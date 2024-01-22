import onnx
import torch
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx, fuse_fx
from torch.ao.quantization import QConfigMapping
import edgeai_torchmodelopt


#define the model
class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,8,3)

    def forward(self, x):
        x = self.conv1(x)
        return x


example_model = ExampleModel()


# export onnx
example_model.eval()
example_input = torch.rand((1, 3,32,32))
torch.onnx.export(example_model, example_input, 'example_model_float.onnx')

total_epochs = 10

# quantize with ptq
# qconfig_mapping = torch.ao.quantization.get_default_qat_qconfig_mapping()
# # qconfig_settings = torch.ao.quantization.default_per_channel_symmetric_qnnpack_qconfig
# # qconfig_mapping = QConfigMapping().set_global(qconfig_settings)
# prepared_model = prepare_qat_fx(example_model, qconfig_mapping, example_input)

# quantize with edgeai_torchmodelopt wrapper
# qconfig_type = edgeai_torchmodelopt.xmodelopt.quantization.v2.qconfig.QConfigType.WC8SYM_AT8SYM
qconfig_type = edgeai_torchmodelopt.xmodelopt.quantization.v2.qconfig.QConfigType.WC8SYMP2_AT8SYMP2
prepared_model = edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule(example_model, qconfig_type=qconfig_type, total_epochs=total_epochs)

# calibration
for it in range(total_epochs):
    data_input = torch.rand((1, 3,32,32))
    prepared_model(data_input)

quantized_model = prepared_model.convert()
torch.onnx.export(quantized_model, example_input, 'example_model_qdq.onnx')


# Convert QDQ format to Int8 format
import onnxruntime as ort
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
so.optimized_model_filepath = 'example_model_int.onnx'
# logger.info("Inplace conversion of QDQ model to INT8 model at: {}".format(onnx_file))
ort.InferenceSession('example_model_qdq.onnx', so)
