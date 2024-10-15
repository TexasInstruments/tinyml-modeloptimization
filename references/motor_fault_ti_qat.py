# torch imports
import test
import torch
import torch.utils
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split

# ti, onnx imports
import tinyml_torchmodelopt.quantization as tinpu_quantization  # type: ignore
import onnx
import onnxruntime as ort

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class MotorFaultDataset(Dataset):
    def __init__(self, X, Y):
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.y = torch.from_numpy(Y).type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


def get_dataset_from_csv(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    Y = df['Target'].to_numpy()
    for column in df.columns:
        df[column] = df[column] / df[column].abs().max()
    X = df[['Vibx', 'Viby', 'Vibz']].to_numpy()
    return X[:10000], Y[:10000]


def get_dataloader_from_dataset(X, Y):
    dataset = MotorFaultDataset(X, Y)
    train_dataset, test_dataset = random_split(dataset, [0.75, 0.25])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32)
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer):
    avg_loss = 0
    model.train()
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += float(loss)
    avg_loss = avg_loss/len(dataloader)
    return avg_loss, model, loss_fn, optimizer


def get_nn_model(n_in, h1, h2, n_out):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.layer1 = nn.Linear(n_in, h1)
            self.layer2 = nn.Linear(h1, h2)
            self.layer3 = nn.Linear(h2, n_out)

        def forward(self, x):
            """Specify how data passes through the network."""
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            x = self.layer3(x)
            return x
    nn_model = NeuralNetwork().to(DEVICE)
    return nn_model


def get_qat_model(nn_model):
    qconfig_type = {
        'weight': {
            'bitwidth': 8,
            'qscheme': torch.per_channel_symmetric,
            'power2_scale': True
        },
        'activation': {
            'bitwidth': 8,
            'qscheme': torch.per_tensor_symmetric,
            'power2_scale': True,
            'range_max': None,
            'fixed_range': False
        }
    }

    # Define the QAT model
    QAT_model = tinpu_quantization.TINPUTinyMLQATFxModule(
        nn_model, qconfig_type=qconfig_type, total_epochs=5)

    return QAT_model


def train_model(model, dataloader):
    loss_fn = torch.nn.CrossEntropyLoss()
    opti = torch.optim.Adam(params=model.parameters(), lr=0.008)
    for epoch in range(5):
        loss, model, loss_fn, opti = train(dataloader, model, loss_fn, opti)
        print(f"Epoch:  {epoch+1}  Loss :  {round(loss, 5)}")
    return model


def export_QAT_model(QAT_model, model_name):

    def rename_input_node_for_onnx_model(onnx_model, input_node_name):
        """Rename the node of an ONNX model"""
        # Update graph input name.
        onnx_model.graph.input[0].name = input_node_name
        # Update input of the first node also to correspond.
        onnx_model.graph.node[0].input[0] = input_node_name
        # Check and write out the updated model
        onnx.checker.check_model(onnx_model)
        return onnx_model

    # Convert PyTorch QDQ layers to TI NPU int8 layers.
    QAT_model.to(DEVICE)
    QAT_model = QAT_model.convert()

    # Export int8 quantized model to onnx.
    dummy = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    QUANTIZED_MODEL = f"{model_name}"
    QAT_model.export(dummy, QUANTIZED_MODEL)

    # Set input name in the ONNX model to 'input' for consistency with float model
    load_onnx = onnx.load(QUANTIZED_MODEL)
    updated_model = rename_input_node_for_onnx_model(load_onnx, 'input')
    onnx.save(updated_model, QUANTIZED_MODEL)
    return updated_model


def validate_saved_model(model_name, dataloader):
    correct_predictions = 0
    total_predictions = 0
    ort_session_options = ort.SessionOptions()
    ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort_session = ort.InferenceSession(f"{model_name}", ort_session_options)

    for _, (X, Y) in enumerate(dataloader):
        for idx in range(len(X)):
            total_predictions += 1
            data_point = X[idx].numpy()
            outputs = ort_session.run(None, {"input": data_point})
            conf_1 = outputs[0][0][0][0]
            if conf_1.argmax(0) == Y[idx]:
                correct_predictions += 1
    accuracy = round(correct_predictions*100/total_predictions,5)
    return accuracy

def validate_qat_model(model, test_loader):
    import pandas as pd
    import numpy as np

    y_pred = model(test_loader.dataset.dataset.x)
    _, y_pred = torch.max(y_pred.data[0][0], 1)
    eval_matrix_dropout = (pd.crosstab(test_loader.dataset.dataset.y, y_pred))
    print("Confusion Matrix")
    print(eval_matrix_dropout)

    # Accuracy of the model
    accuracy = np.diag(eval_matrix_dropout).sum()/test_loader.dataset.dataset.y.shape[0]
    print("Accuracy: ", round(accuracy, 5))
    return accuracy

if __name__ == '__main__':

    MODEL_NAME = "data/motor_fault.onnx"
    CSV_FILE = "data/dataset.csv"

    X, Y = get_dataset_from_csv(CSV_FILE)
    train_loader, test_loader = get_dataloader_from_dataset(X, Y)

    nn_model = get_nn_model(n_in=3, h1=24, h2=12, n_out=4)
    nn_model = train_model(nn_model, train_loader)

    qat_model = get_qat_model(nn_model)
    qat_model = train_model(qat_model, train_loader)

    export_QAT_model(qat_model, MODEL_NAME)
    accuracy = validate_qat_model(qat_model, test_loader)
    accuracy = validate_saved_model(MODEL_NAME, test_loader)
    print(accuracy)
    exit()
