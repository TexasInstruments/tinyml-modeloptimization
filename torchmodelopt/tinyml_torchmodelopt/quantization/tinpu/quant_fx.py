#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################

import platform
import torch
import types
import operator

from torch.fx import GraphModule
from typing import List, Tuple

import edgeai_torchmodelopt

from ..common import TinyMLQConfigFormat, GenericTinyMLQATFxModuleBase
from . import quant_utils

def are_both_function_equal(first_function, second_function) -> bool:

    operationDict = {torch.add: operator.add,torch.sub: operator.sub,torch.mul: operator.mul,
                        operator.add: torch.add,operator.sub: torch.sub,operator.mul: torch.mul}
    if first_function == second_function:
        return True
    elif hasattr(first_function, 'target') and first_function.target in operationDict.keys():
        # if it is one  of add, sub, mul from either of operator module or torch module it should be the counter part
        return second_function == operationDict[first_function]
    elif first_function in operationDict.keys():
        # if it is one  of add, sub, mul from either of operator module or torch module it should be the counter part
        return second_function == operationDict[first_function]
    else:
        return False
    

def simple_chain_searcher(main_module: GraphModule, pattern_type: List) -> List[torch.Node]:

    main_module_nodes = list(main_module.graph.nodes)
    main_module_length = len(main_module_nodes)
    pattern_type_length = len(pattern_type)

    assert isinstance(pattern_type, list) and all(isinstance(typ, (type, str,)) or isinstance(typ, (types.FunctionType, types.BuiltinFunctionType)) for typ in pattern_type), \
        'This function only supports searching for a straight sequence of types of module!'

    main_module_idx = 0
    pattern_type_idx = 0

    matched_patterns = list()
    next_match = -1

    inp, out = None, None

    def is_both_node_equal(main_module_node: torch.Node, pattern_type_node: torch.Node) -> bool:
        both_node_equal = main_module_node.op == 'call_module' and isinstance(pattern_type_node, type) and isinstance(dict(main_module.named_modules())[main_module_node.target], pattern_type_node)
        both_node_equal = both_node_equal or (main_module_node.op == 'call_method' and isinstance(pattern_type_node, str) and main_module_node.target == pattern_type_node)
        both_node_equal = both_node_equal or (main_module_node.op == 'call_function' and isinstance(pattern_type_node, (types.FunctionType, types.BuiltinFunctionType)) and are_both_function_equal(main_module_node.target, pattern_type_node))
        both_node_equal = both_node_equal or (main_module_node.op == 'placeholder' and isinstance(pattern_type_node, str) and main_module_node.target == pattern_type_node)
        return both_node_equal
    
    while (main_module_idx < main_module_length):

        main_module_node = main_module_nodes[main_module_idx]
        pattern_type_node = pattern_type[pattern_type_idx]
        # Check if both main module node and pattern node are equal or not
        both_node_equal = is_both_node_equal(main_module_node, pattern_type_node)
        if both_node_equal:
            if main_module_node == pattern_type[0] and next_match == -1 and pattern_type_idx != 0:
                # if another pattern is matching inside the current matching pattern
                next_match = main_module_idx
            if pattern_type_idx == 0:
                # Node is matched with 1st node of pattern
                inp = main_module_node

            main_module_idx += 1
            pattern_type_idx += 1
            if pattern_type_idx == pattern_type_length:
                # Append the nodes which matched the pattern
                out = main_module_node
                matched_patterns.append((inp, out))
                next_match = -1
        else:
            # Reset the values as nodes didn't match
            inp, out = None, None
            pattern_type_idx = 0
            if next_match == -1:
                main_module_idx += 1
            else:
                main_module_idx = next_match
                next_match = -1
        pattern_type_idx = pattern_type_idx % pattern_type_length

    return matched_patterns


class TINPUTinyMLQATFxModule(GenericTinyMLQATFxModuleBase):
    def __init__(self, *args, qconfig_type=None, **kwargs) -> None:
        if qconfig_type is None:
            # there are multiple ways to specify qconfig_type - one is to use a dictionary like this.
            # qconfig_type = qconfig_type or dict(weight=dict(bitwidth=8, qscheme=torch.per_channel_symmetric, power2_scale=True),
            #   activation=dict(bitwidth=8, qscheme=torch.per_tensor_symmetric, power2_scale=True, range_max=None, fixed_range=False))
            # another way is to use one of the predefined presets
            qconfig_type = edgeai_torchmodelopt.xmodelopt.quantization.v2.qconfig_types.QConfigType.WC8SYMP2_AT8SYMP2
        backend = 'fbgemm' if platform.system() in ['Windows'] else 'qnnpack'
        super().__init__(*args, qconfig_type=qconfig_type, backend=backend, **kwargs)

    def convert(self, *args, backend_config=None, model_qconfig_format=TinyMLQConfigFormat.TINPU_INT_MODEL, output_dequantize=False, **kwargs):
        # first convert the model to int
        from torch.ao.quantization.backend_config import get_native_backend_config, BackendPatternConfig, ObservationType, DTypeConfig
        backend_config = get_native_backend_config()
        weighted_int8_dtype_config = DTypeConfig(
            input_dtype=torch.quint8,
            output_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            bias_dtype=torch.float)
        flatten_config = BackendPatternConfig(torch.nn.modules.flatten.Flatten) \
            .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT) \
            .add_dtype_config(weighted_int8_dtype_config) \
            .set_root_module(torch.nn.modules.flatten.Flatten) \
            .set_qat_module(torch.nn.modules.flatten.Flatten) \
            .set_reference_quantized_module(torch.nn.modules.flatten.Flatten)
        backend_config = backend_config.set_backend_pattern_config(flatten_config)
        backend_config = None
        super().convert(*args, model_qconfig_format=model_qconfig_format, backend_config=backend_config, **kwargs)
        _convert_replacement_func = lambda module, pattern, *largs, **lkwargs: self._convert_replacement(module, pattern, *largs, output_dequantize=output_dequantize, **lkwargs)
        # then apply the transformation to required output format
        if model_qconfig_format == TinyMLQConfigFormat.TINPU_INT_MODEL:
            self.module = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(self.module, replacement_dict={'tinyml_modelopt_quant_replace_types': {'quant_replace_types': _convert_replacement_func}})

        return self

    def export(self, *args, model_qconfig_format=TinyMLQConfigFormat.TINPU_INT_MODEL, simplify=True, skipped_optimizers=None, **kwargs):
        skipped_optimizers = skipped_optimizers or ['fuse_add_bias_into_conv', 'eliminate_nop_with_unit']
        super().export(*args, model_qconfig_format=model_qconfig_format, simplify=simplify, skipped_optimizers=skipped_optimizers, **kwargs)

    def measure_stats(self, float_output, quant_output):
        diff_output = (float_output - quant_output)
        diff_output_abs = diff_output.abs()
        diff_output_sqr = diff_output**2
        float_output_sqr = float_output**2
        quant_error_min = diff_output_abs.min().item()
        quant_error_max = diff_output_abs.max().item()
        quant_error_mean = diff_output_abs.mean().item()
        quant_snr_db = (10 * torch.log10(float_output_sqr.mean() / diff_output_sqr.mean())).item()
        quant_psnr_db = (10 * torch.log10(float_output_sqr.max() / diff_output_sqr.mean())).item()
        quant_absmu_by_sigma = (float_output.abs().mean() / diff_output.std()).item()
        diff_output_stats = dict(snr_db=quant_snr_db, psnr_db=quant_psnr_db, absmu_by_sigma=quant_absmu_by_sigma,
                                 mean=quant_error_mean, min=quant_error_min, max=quant_error_max)
        return diff_output_stats

    def is_batch_normalized(self, module: GraphModule) -> bool:
        named_modules = dict(module.named_modules())
        batch_norm_modules = (torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d, torch.nn.BatchNorm2d)
        for name_entry, module_entry in named_modules.items():
            if len(list(module_entry.parameters(recurse=False))) > 0 and isinstance(module_entry, batch_norm_modules):
                return True
        return False
    
    def get_scales_of_nodes(module: GraphModule, scales_of_nodes: List[float]) -> List[float]:
        
        def recur_args(node: torch.Node) -> float:
            bfs = [arg for arg in node.args]
            for node in bfs:
                if str(node) in scales_of_nodes.keys():
                    return scales_of_nodes[str(node)]
                else:
                    bfs += [arg for arg in node.args]
                bfs.pop(0)
            return None

        nodes = list(module.graph.nodes)
        for node in nodes:
            if str(node) not in scales_of_nodes.keys():
                scales_of_nodes[str(node)] = recur_args(node)

        return scales_of_nodes

    def replacement_rules(self, is_batch_normalized: bool, output_dequantize: bool) -> List[Tuple]:

        replacement_rules = []
        # Batch Normalization Modules
        if is_batch_normalized:
            replacement_rules = [([torch.quantize_per_tensor, torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d], quant_utils.TINPUQuantizedReplacement.from_q_qbn)]
        else:
            replacement_rules = [([torch.quantize_per_tensor, torch.nn.Identity], quant_utils.TINPUQuantizedReplacement.from_q_id),]

        replacement_rules = replacement_rules + [
            ([torch.nn.Sequential],quant_utils.TINPUQuantizedReplacement.from_child_module),
            ([torch.nn.Module], quant_utils.TINPUQuantizedReplacement.from_child_module),
            # Pooling Modules
            ([torch.nn.AvgPool2d], quant_utils.TINPUQuantizedReplacement.from_passthrough_module),                   # OSS required
            ([torch.nn.AdaptiveAvgPool2d], quant_utils.TINPUQuantizedReplacement.from_adaptiveavgpool2d),            # OSS required
            ([torch.nn.MaxPool2d], quant_utils.TINPUQuantizedReplacement.from_passthrough_module),                   # OSS not required
            # Flatten Modules
            (['dequantize', torch.nn.Flatten], quant_utils.TINPUQuantizedReplacement.from_dq_flatten),               # Removes quantization
            (['x', torch.nn.Flatten], quant_utils.TINPUQuantizedReplacement.from_x_flatten),                         # Replaces quantization
            # ConvRelu2D Module
            ([torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d], quant_utils.TINPUQuantizedReplacement.from_qconv_relu),
            # LinearRelu Module
            ([torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearReLU], quant_utils.TINPUQuantizedReplacement.from_qlinear_relu),
            # Linear Module
            ([torch.ao.nn.quantized.modules.linear.Linear], quant_utils.TINPUQuantizedReplacement.from_qlinear),
        ]
        # Dequantization Module
        if output_dequantize:
            # Replaces dequantization layer with OSS
            replacement_rules += [(['dequantize'], quant_utils.TINPUQuantizedReplacement.from_dq_with_dq)]
        else:
            # Replaces dequantization layer with Identity
            replacement_rules += [(['dequantize'], quant_utils.TINPUQuantizedReplacement.from_dq)]

        return replacement_rules

    def _convert_replacement(self, module: GraphModule, pattern, *args, output_dequantize: bool = False, **kwargs) -> GraphModule:
        # Check if the model has batch normalization
        is_batch_normalized = self.is_batch_normalized(module)
        # Convert the module using symbolic trace
        module = torch.fx.symbolic_trace(module) if not isinstance(module, torch.fx.GraphModule) else module
        # Get the replacement rules to change the pattern
        replacement_rules = self.replacement_rules(is_batch_normalized, output_dequantize)
        # Give each module a unique module_no
        module_no = 0
        # Replace the patterns using the replacement function
        # print(list(module.graph.nodes))
        for replacement_pattern, replacement_function in replacement_rules:
            matches = simple_chain_searcher(module, replacement_pattern)
            for (start, end) in matches:
                replacement_function(module, start, end, module_no)
                module_no += 1
                # print(list(module.graph.nodes))
        return module
