import numpy as np
import torch
import model_quant_utils

#######################################################################################
opset_version = 17


#######################################################################################
# model definition and export (float)
class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = torch.nn.BatchNorm2d(3)
        # self.bn0.with_convert_custom_config = True

        self.conv1 = torch.nn.Conv2d(3,8,3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


example_model = ExampleModel()

#######################################################################################
# model training should go in here.
# for this demonstration we are using the untrained model with random parameters.
#######################################################################################


#######################################################################################
# export onnx
example_model.eval()
example_input = torch.rand((1, 3,32,32))
torch.onnx.export(example_model, example_input, 'example_model_float.onnx', opset_version=opset_version)


#######################################################################################
# quantization
# to install this package, from the torchmodelopt folder do,
# pip install -e ./
# in the following repository
# https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-modeloptimization/browse
import edgeai_torchmodelopt

total_epochs = 10

# quantize with ptq
# qconfig_mapping = torch.ao.quantization.get_default_qat_qconfig_mapping()
# # qconfig_settings = torch.ao.quantization.default_per_channel_symmetric_qnnpack_qconfig
# # qconfig_mapping = QConfigMapping().set_global(qconfig_settings)
# prepared_model = prepare_qat_fx(example_model, qconfig_mapping, example_input)


##################################################################################
# quantize with edgeai_torchmodelopt wrapper
# qconfig_type = edgeai_torchmodelopt.xmodelopt.quantization.v2.qconfig.QConfigType.WC8SYMP2_AT8SYMP2
qconfig_type = dict(weight=dict(bitwidth=8, qscheme=torch.per_channel_symmetric, power2_scale=True),
                    activation=dict(bitwidth=8, qscheme=torch.per_tensor_symmetric, power2_scale=True, range_max=None, fixed_range=False))
prepared_model = edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule(example_model, qconfig_type=qconfig_type, total_epochs=total_epochs)


##################################################################################
# here we use Post-Training-Quantization (PTQ), but we can use QAT as well.
for it in range(total_epochs):
    data_input = torch.rand((1, 3,32,32))
    prepared_model(data_input)

torch.onnx.export(prepared_model, example_input, 'example_model_fakeq.onnx', opset_version=opset_version)


##################################################################################
# TODO: remove
# native pytorch int8 quantization
# convert_custom_config = model_export_torch_utils.get_convert_custom_config()
# backend_config = torch.ao.quantization.backend_config.get_native_backend_config()
# quantized_model = prepared_model.convert()
# try:
#     torch.onnx.export(quantized_model, example_input, 'example_model_qdq.onnx', opset_version=opset_version)
# except:
#     print('ERROR: converted qdq model could not be exported to onnx')

##################################################################################
# # Convert QDQ format to Int8 format
# import onnxruntime as ort
# so = ort.SessionOptions()
# so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# so.optimized_model_filepath = 'example_model_int.onnx'
# # logger.info("Inplace conversion of QDQ model to INT8 model at: {}".format(onnx_file))
# ort.InferenceSession('example_model_qdq.onnx', so)


##################################################################################
def quantize_replacement_function(model, pattern, *args, **kwargs):
    model = torch.fx.symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model

    # replace BN
    pattern_list = [edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize, torch.nn.BatchNorm2d, edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize]
    matches = edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer.straight_type_chain_searcher(model, pattern_list)
    for no_of_module_replaced, (start, end) in enumerate(matches):
        fq_module1 = dict(model.named_modules())[start.target]
        bn_module = dict(model.named_modules())[start.next.target]
        fq_module2 = dict(model.named_modules())[end.target]
        new_fq_module = model_quant_utils.OffsetScaleShift.from_fq_bn_fq(fq_module1, bn_module, fq_module2)
        edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer._replace_pattern(model, start, end, new_fq_module, no_of_module_replaced)
    #

    # replace torch.ao.nn.intrinsic.qat.ConvBnReLU2d
    pattern_list = [torch.ao.nn.intrinsic.qat.ConvBnReLU2d, edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize]
    matches = edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer.straight_type_chain_searcher(model, pattern_list)
    for no_of_module_replaced, (start, end) in enumerate(matches):
        cbn_module = dict(model.named_modules())[start.target]
        fq_module = dict(model.named_modules())[end.target]
        new_fq_module = model_quant_utils.OffsetScaleShift.from_cbn_fq(cbn_module, fq_module)
        edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer._replace_pattern(model, start, end, new_fq_module, no_of_module_replaced)
    #

    return model

replacement_dict = {
    'replace_types1': quantize_replacement_function
}
prepared_model.module = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(prepared_model.module, replacement_dict)

##################################################################################
try:
    torch.onnx.export(prepared_model, example_input, 'example_model_tinie.onnx', opset_version=opset_version)
except:
    print('ERROR: converted qdq tinie model could not be exported to onnx')
