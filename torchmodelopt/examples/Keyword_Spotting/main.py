# torch imports
from edgeai_torchmodelopt import QuantizationVersion
from torch.ao.quantization import quantize_fx
import torch.utils
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split
import torchinfo
import os

from get_dataset_torch import SavedTensorDataset
import torch_model as models
# ti, onnx imports
from tinyml_torchmodelopt.quantization import \
    TINPUTinyMLQATFxModule, TINPUTinyMLPTQFxModule, GenericTinyMLQATFxModule, GenericTinyMLPTQFxModule
import onnx
import onnxruntime as ort
import sys
# other imports
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.metrics import confusion_matrix
import torch.ao.ns._numeric_suite_fx as ns
from torch.fx import symbolic_trace
from torch.ao.quantization.observer import FixedQParamsObserver
import torch.ao.quantization as quantization


import torch
import torch.nn.functional as F
import get_dataset_torch as kws_data
import kws_util
from torchinfo import summary
from torchmetrics.classification import Accuracy
from tqdm import tqdm
import tensorflow as tf
import tarfile
import torch.optim as optim


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(dataloader, model, loss_fn, optimizer, scheduler, device="cpu"):
    """
    Train the model for one epoch and print running accuracy.
    """
    model.train()
    
    first_batch = next(iter(dataloader), None)
    if first_batch is None:
        print("🚨 Error: No data in dataloader! Exiting training loop.")
        return 0, 0, model, loss_fn, optimizer
    
    print(" First batch keys:", first_batch.keys())  
    print(" First batch shapes:", first_batch["audio"].shape, first_batch["label"].shape)  

    total_loss = 0
    running_corrects = 0
    total_samples = 0

    accuracy_metric = Accuracy(task="multiclass", num_classes=12).to(device)  # Adjust `num_classes` as needed

    for batch_idx, data in enumerate(dataloader):
        inputs, targets = data['audio'].to(device), data['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        batch_acc = accuracy_metric(preds, targets)
        
        # Running statistics
        total_loss += loss.item()
        running_corrects += torch.sum(preds == targets).item()
        total_samples += targets.size(0)

        # Print running accuracy every 10 batches
        if batch_idx % 10 == 0:
            print(f"🟢 Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}, Running Accuracy: {batch_acc:.4f}")

    # Compute final epoch loss & accuracy
    avg_loss = total_loss / len(dataloader)
    avg_acc = running_corrects / total_samples  # Overall epoch accuracy
    if scheduler:
        scheduler.step()
    print(f" Epoch Finished - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")

    return avg_loss, avg_acc, model, loss_fn, optimizer

def train_model(model, train_loader, total_epochs, learning_rate, device="cpu"):
    """
    Train the model for multiple epochs and display accuracy.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=kws_util.lr_schedule(learning_rate))

    for epoch in range(total_epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{total_epochs}")

        loss, acc, model, loss_fn, optimizer = train(train_loader, model, loss_fn, optimizer, scheduler, device)

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        print(f"📌 Epoch {epoch+1} - LR: {last_lr:.5f} - Loss: {loss:.5f} - Accuracy: {acc:.4f}")

    return model
# Validation & Testing Function
def test(model, test_loader, loss_fn, device="cpu"):
    """
    Evaluate the model on test or validation data.
    
    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test/validation dataset.
        loss_fn: Loss function.
        device: Device to evaluate on.
    
    Returns:
        test_loss: Average test loss.
        test_acc: Average test accuracy.
    """
    model.eval()
    test_loss = 0
    test_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=12).to(device)

    with torch.inference_mode():
        for batch in test_loader:
            inputs, targets = batch['audio'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            _, preds = torch.max(F.softmax(outputs, dim=1), 1)
            test_loss += loss.item()
            test_acc += accuracy(preds, targets.squeeze()).item()

    return test_loss / len(test_loader), test_acc / len(test_loader)

def calibrate(dataloader: DataLoader, model: nn.Module, loss_fn):
    """
    Calibrate the model using the provided dataloader.
    Compute the loss for the purpose of information.
    Returns the average loss.
    """
    model.train()  # Ensure the model is in evaluation mode
    avg_loss = 0.0
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
            inputs = batch["audio"].to(DEVICE)
            labels = batch["label"].clone().to(torch.long).to(DEVICE)
          #  labels = torch.tensor(batch["label"], dtype=torch.long).to(DEVICE)  # Convert to tensor and move to device
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.flatten(start_dim=1)

            # Compute the loss
            loss = loss_fn(outputs, labels)
            avg_loss += loss.item()

          #  print(f"Calibration Batch {batch_idx}: Loss: {loss.item()}")

    avg_loss /= total_batches
    print(f"Calibration complete. Average Loss: {avg_loss}")
    return avg_loss, model, loss_fn, None

def get_quant_model(nn_model: nn.Module, example_input: torch.Tensor, total_epochs: int, weight_bitwidth: int,
        activation_bitwidth: int, quantization_method: str, quantization_device_type: str) -> nn.Module:
    """
    Convert the torch model to quant wrapped torch model. The function requires
    an example input to convert the model.
    """

    '''
    The QAT wrapper module does the preparation like in:
    quant_model = quantize_fx.prepare_qat_fx(nn_model, qconfig_mapping, example_input)
    It also uses an appropriate qconfig that imposes the constraints of the hardware.

    The api being called doesn't actually pass qconfig_type - so it will be defined inside. 
    But if you need to pass, it can be defined.
    '''
    if weight_bitwidth is None or activation_bitwidth is None:
        '''
        # 8bit weight / activation is default - no need to specify inside.
        qconfig_type = {
            'weight': {
                'bitwidth': 8,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
            },
            'activation': {
                'bitwidth': 8,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
            }
        }
        '''
        qconfig_type = None
    elif weight_bitwidth == 8:
        qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': True,
                'range_max': None,
                'fixed_range': False
            }
        }
    elif weight_bitwidth == 4:
        qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False
            }
        }
    elif weight_bitwidth == 2:
        qconfig_type = {
            'weight': {
                'bitwidth': weight_bitwidth,
                'qscheme': torch.per_channel_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False,
                'quant_min': -1,
                'quant_max': 1,
            },
            'activation': {
                'bitwidth': activation_bitwidth,
                'qscheme': torch.per_tensor_symmetric,
                'power2_scale': False,
                'range_max': None,
                'fixed_range': False
            }
        }
    else:
        raise RuntimeError("unsupported quantization parameters")
    #
    
#     # ✅ Step 1: Collect all layers that need BatchNorm
#     missing_bn_layers = []
#     for name, module in nn_model.named_modules():
#      if isinstance(module, nn.Conv2d) and not hasattr(module, "bn"):
#         print(f"Adding BatchNorm2d to {name}")
#         missing_bn_layers.append((name, module.out_channels))

# # ✅ Step 2: Apply the changes after iteration
#     for name, out_channels in missing_bn_layers:
#      setattr(nn_model, name + "_bn", nn.BatchNorm2d(out_channels))

#     for name, module in nn_model.named_modules():
#      if "weight_fake_quant" in name and getattr(module, "qconfig", None) is None:
#         setattr(module, "qconfig", quantization.get_default_qconfig("fbgemm"))
#      fixed_qparams_qconfig = quantization.QConfig(
#     activation=FixedQParamsObserver.with_args(scale=1.0 / 255, zero_point=0, dtype=torch.quint8),
#     weight=torch.ao.quantization.default_per_channel_weight_observer
#     )
    
# # Apply `FixedQParamsObserver` only to ReLU, Softmax, and AvgPool layers
#    qconfig_mapping =quantization.QConfigMapping()
#     for name, module in nn_model.named_modules():
#      print("modifying")
#      if "activation_" in name or "dense_softmax" in name or "average_pooling2d" in name:
#         qconfig_mapping.set_module_name(name, fixed_qparams_qconfig)

#     for name, module in nn_model.named_modules():
#      if "activation_post_process" in name and getattr(module, "qconfig", None) is None:
#         print(f"Applying FixedQParamsObserver to {name}")
#         qconfig_mapping.set_module_name(name, fixed_qparams_qconfig)
   
    if quantization_device_type == 'TINPU':
        if quantization_method == 'QAT':
            quant_model = TINPUTinyMLQATFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        elif quantization_method == 'PTQ':
            quant_model = TINPUTinyMLPTQFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        else:
            raise RuntimeError(f"Unknown Quantization method: {quantization_method}")
        #
    elif quantization_device_type == 'GENERIC':
        if quantization_method == 'QAT':
            quant_model = GenericTinyMLQATFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        elif quantization_method == 'PTQ':
            quant_model = GenericTinyMLPTQFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
        else:
            raise RuntimeError(f"Unknown Quantization method: {quantization_method}")
        #
    else:
        raise RuntimeError(f"Unknown Quantization device type: {quantization_device_type}")

    
    return quant_model

def calibrate_model(model: nn.Module, dataloader: DataLoader, total_epochs: int) -> nn.Module:
    """
    Calibrate the model for PTQ - (torch model or qat wrapped torch model) with the given train dataloader,
    learning_rate and loss are not needed for PTQ / calibration as backward / back propagation is not performed.
    loss_fn is used here only for the purpose of information - to know how good is the calibration.
    """
    # loss_fn for multi class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
     for epoch in range(total_epochs):
        # train the model for an epoch
        loss, model, loss_fn, opti = calibrate(dataloader, model, loss_fn)
        last_lr = 0
       # print(f"Epoch: {epoch+1}\t LR: {round(last_lr,5)}\t Loss: {round(loss, 5)}")

    return model

def rename_input_node_for_onnx_model(onnx_model, input_node_name: str):
    """Rename the node of an ONNX model"""
    # Update graph input name.
    onnx_model.graph.input[0].name = input_node_name
    # Update input of the first node also to correspond.
    onnx_model.graph.node[0].input[0] = input_node_name
    # Check and write out the updated model
    onnx.checker.check_model(onnx_model)
    return onnx_model


def export_model(quant_model, example_input: torch.Tensor, model_name: str, with_quant: bool = False):
    """
    Export the quantized model and print its layer-wise quantization parameters.
    """

    quant_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert model using FX Graph-based quantization if needed
    if with_quant:
        if hasattr(quant_model, "convert"):
         print(" Running `convert()` on quant_model...")
         quant_model = quant_model.convert()
       # compare_fp32_and_quantized_weights(quant_model)
        else:
         quant_model = quantize_fx.convert_fx(quant_model.module)
    #  Print layer-wise quantization parameters with range
  #  print_layerwise_quant_params(quant_model)
   # print(quant_model)
   
    #  Save quantized model
    q_model_path = "final_quantized_model.pth"
    torch.save(quant_model.state_dict(), q_model_path)
    print(f" Quantized final model saved at {q_model_path}")

    #  Print final model structure to file
    file_path = "final_model_print.txt"
    with open(file_path, "w") as f:
        f.write(str(quant_model))

    #  Export to ONNX
    if hasattr(quant_model, "export"):
        print(" Exporting to ONNX...")
        quant_model.export(example_input, model_name, input_names=['input'])
    else:
        torch.onnx.export(quant_model, example_input, model_name, input_names=['input'])

    print(" Model exported successfully")
    return quant_model


def validate_model(model: nn.Module, test_loader: DataLoader, num_categories: int, categories_name: List[str]) -> float:
    """
    The function takes the model (torch model or qat wrapped torch model), torch dataloader
    and the num_categories to give the confusion matrix and accuracy of the model.
    """
    model.eval()
    y_target = []
    y_pred = []

    for batch_idx, batch in enumerate(test_loader):
      #   X, y =  batch["audio"].clone.to(DEVICE), batch["label"].clone().to(torch.long).to(DEVICE)
        X = batch["audio"].clone().to(DEVICE)
        y = batch["label"].clone().to(torch.long).to(DEVICE)
    #    X, y = batch["audio"].to(DEVICE).float(), torch.tensor(batch["label"], dtype=torch.long).to(DEVICE)
        # make prediction for the current batch
        pred = model(X)
        pred = pred.flatten(start_dim=1)
        # take the max probability among the classes predicted
        _, pred = torch.max(pred, 1)
        y_pred.append(pred.numpy())
        y_target.append(y.numpy())

    y_pred = np.concatenate(y_pred)
    y_target = np.concatenate(y_target)
    categories_idx = np.arange(0, num_categories, 1)
    # create a confusion matrix
    cf_matrix = confusion_matrix(y_target, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[categories_name[i] for i in categories_idx],
                         columns=[categories_name[i] for i in categories_idx])
    print()
    print("Confusion Matrix")
    print(df_cm)

    # Accuracy of the model
    accuracy = np.diag(df_cm).sum()/np.array(df_cm).sum()
    return accuracy


def validate_saved_model(model_name: str, dataloader: DataLoader) -> float:
    """
    The function takes the saved onnx model, torch test dataloader to give the accuracy of the model.
    """
    correct_predictions = 0
    total_predictions = 0
    # set ort inference session options
    ort_session_options = ort.SessionOptions()
    ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # start the inference session with the model_name saved in disk
    ort_session = ort.InferenceSession(f"{model_name}", ort_session_options)
    

    for _, batch in enumerate(dataloader):
     X= batch["audio"].clone().to(DEVICE)
     Y= batch["label"].clone().to(torch.long).to(DEVICE)
  #   X, Y = batch["audio"], batch["label"]
     for idx in range(len(X)):
            total_predictions += 1
            # add a new axis at the beginning of data
            data_point = X[idx].numpy()[np.newaxis, ...]
            # classify the data_point
            # outputs = ort_session.run(None, {'x': data_point})
            outputs = ort_session.run(None, {'input': data_point})
            conf_1 = outputs[0].flatten()
            # check if the classification is correct
            if conf_1.argmax(0) == Y[idx]:
                correct_predictions += 1

    accuracy = round(correct_predictions/total_predictions, 5)
    return accuracy



def load_calibration_indices(file_path):
    """Load indices from a calibration indices file."""
    with open(file_path, "r") as f:
        indices = [int(line.strip()) for line in f if line.strip().isdigit()]
    return indices

def create_calibration_dataset(dataset, indices):
    """Create a calibration dataset using specified indices."""
    from torch.utils.data import Subset
    return Subset(dataset, indices)


    
if __name__ == '__main__':

    MODEL_NAME = "kws.onnx"
    CATEGORIES_NAME = [ 0,1,2,3,4,5,6,7,8,9,10,11]
    NUM_EPOCHS = 36 #10 acc was 92.2
    WINDOW_LENGTH = 1024
    WINDOW_OFFSET = WINDOW_LENGTH//4  # WINDOW_LENGTH//2
    LEARNING_RATE = 0.00001
    QUANTIZATION_METHOD = 'PTQ' #'PTQ' #'QAT' #None
    WEIGHT_BITWIDTH = 8 #2 #4 #8
    ACTIVATION_BITWIDTH = 8 #8 #4 #2
    QUANTIZATION_DEVICE_TYPE = 'TINPU' #'TINPU', 'GENERIC'
    NORMALIZE_INPUT = True #True, #False

    assert QUANTIZATION_DEVICE_TYPE != 'GENERIC' or (not NORMALIZE_INPUT), \
        'normalizing input with BatchNorm is not supported for the export format used for Generic Quantization. Please set NORMALIZE_INPUT to False.'

    
    Flags, unparsed = kws_util.parse_command()
#    train_loader = SavedTensorDataset(dataset_dir=r"C:\Users\A0507182\torch_kws_flow\__mlperf_vcdataset\__mlperf_vcdataset\train")
    test_loader = SavedTensorDataset(dataset_dir=r"C:\Users\A0507182\torch_kws_flow\__mlperf_vcdataset\__mlperf_vcdataset\test")
    test_loader = DataLoader(test_loader, batch_size=1, shuffle=False)
 #   train_loader = DataLoader(train_loader, batch_size=100, shuffle=True, num_workers=0)
    
    calibration_indices_file = r"quant_cal_idxs.txt"  # Path to calibration indices

    # Load calibration indices
    calibration_indices = load_calibration_indices(calibration_indices_file)
    print(f"Loaded {len(calibration_indices)} indices for calibration.")
    validation_dataset = SavedTensorDataset(dataset_dir=r"C:\Users\A0507182\torch_kws_flow\__mlperf_vcdataset\__mlperf_vcdataset\val")
    calibration_dataset = create_calibration_dataset(validation_dataset, calibration_indices)
    calibration_loader = DataLoader(dataset=calibration_dataset, batch_size=1, shuffle=False)
    
    example_batch = next(iter(test_loader))


    example_input = example_batch["audio"].float()  # Add channel dimension
 
    nn_model, _ = models.get_model(args=Flags)
  
    #nn_model = nn_model.to("cpu")
  
    #nn_model = train_model(nn_model, train_loader, NUM_EPOCHS, LEARNING_RATE)
    #accuracy = validate_model(nn_model, test_loader, 12, CATEGORIES_NAME)
    #print("OG model accuracy is", accuracy)
    path = r"C:\Users\A0507182\Documents\tinyml-modeloptimization\torchmodelopt\examples\Keyword_Spotting\trained_models\kws_torch.pth"
    checkpoint = torch.load(path , map_location="cpu")
    nn_model.load_state_dict(checkpoint)
    #nn_model=torch.load(path)
    accuracy = validate_model(nn_model, test_loader, 12, CATEGORIES_NAME)
    print("OG model accuracy is", accuracy)
    # ✅ Print final model structure to file
    file_path = "orignal_nn_model_print.txt"
    with open(file_path, "w") as f:
        f.write(str(nn_model))
  
    print("Example Input Shape:", example_input)
    print("Example Input Shape:", example_input.shape)
    example_input = example_input.to(DEVICE).float()

  
    if QUANTIZATION_METHOD in ('QAT', 'PTQ'):
        MODEL_NAME = 'quant_' + MODEL_NAME
        quant_epochs = (NUM_EPOCHS*10) if ((WEIGHT_BITWIDTH<8) or (ACTIVATION_BITWIDTH<8)) else max(NUM_EPOCHS//2, 5)
        quant_model = get_quant_model(nn_model, example_input=example_input, total_epochs=quant_epochs, weight_bitwidth=WEIGHT_BITWIDTH, activation_bitwidth=ACTIVATION_BITWIDTH, quantization_method=QUANTIZATION_METHOD,quantization_device_type=QUANTIZATION_DEVICE_TYPE)
        for name, module in quant_model.named_modules():
         print(f"Layer: {name}, Type: {type(module)}, QConfig: {getattr(module, 'qconfig', None)}")
        if QUANTIZATION_METHOD == 'QAT':
            quant_learning_rate = (LEARNING_RATE/100) if ((WEIGHT_BITWIDTH<8) or (ACTIVATION_BITWIDTH<8)) else (LEARNING_RATE/10)
         
            quant_model = train_model(quant_model, train_loader, quant_epochs, quant_learning_rate)
       
        elif QUANTIZATION_METHOD == 'PTQ':
            quant_model = calibrate_model(quant_model, calibration_loader, quant_epochs)
        quantized_model_path = os.path.join("quantized_model.pth")
        torch.save(quant_model.state_dict(), quantized_model_path)
        print(f"Quantized model saved at {quantized_model_path}")
        accuracy = validate_model(quant_model, test_loader, 12, CATEGORIES_NAME)
        print(f"QAT Model Accuracy: {round(accuracy, 5)}\n")
      
        quant_model = export_model(quant_model, example_input, MODEL_NAME, with_quant=True)

        
    else:
        print("No Quantization method is specified. Will not do quantization.")
    
    accuracy = validate_saved_model(MODEL_NAME, test_loader)
    print(f"Exported ONNX Quant Model Accuracy: {round(accuracy, 5)}")
    path_converted_model=r"C:\Users\A0507182\Documents\tinyml-modeloptimization\torchmodelopt\examples\Keyword_Spotting\final_quantized_model.pth"
    
    nn_model2, _ = models.get_model(args=Flags)
  
    #nn_model = nn_model.to("cpu")
  
    #nn_model = train_model(nn_model, train_loader, NUM_EPOCHS, LEARNING_RATE)
    #accuracy = validate_model(nn_model, test_loader, 12, CATEGORIES_NAME)
    #print("OG model accuracy is", accuracy)
  #  path = r"C:\Users\A0507182\Documents\tinyml-modeloptimization\torchmodelopt\examples\Keyword_Spotting\trained_models\kws_torch.pth"
  #  checkpoint2 = torch.load(path_converted_model , map_location="cpu")
  #  nn_model2.load_state_dict(checkpoint2)
   # accuracy = validate_model(nn_model2, test_loader, 12, CATEGORIES_NAME)
   # print(f"QAT Model Accuracy: {round(accuracy, 5)}\n")
    

