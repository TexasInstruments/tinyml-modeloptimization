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

import copy
import torch
import edgeai_torchmodelopt
from edgeai_torchmodelopt.xmodelopt.quantization.v2 import ModelFormat
from .. import generic
from . import quant_utils


class TINIEModelFormat(ModelFormat):
    TINIE_INT_MODEL = ModelFormat._NUM_FORMATS_
    _NUM_FORMATS_ = ModelFormat._NUM_FORMATS_ + 1


class TINIETinyMLQATFxModule(generic.GenericTinyMLQATFxModule):
    def convert(self, *args, **kwargs):
        super().convert(*args, **kwargs)

    def convert(self, *args, model_format=TINIEModelFormat.TINIE_INT_MODEL, **kwargs):
        self.convert(*args, **kwargs)
        if model_format == TINIEModelFormat.TINIE_INT_MODEL:
            self.module = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(self.module,
                                    replacement_dict={'replace_types1': self._convert_replacement})
        #
        return self

    def export(self, *args, model_format=TINIEModelFormat.TINIE_INT_MODEL, **kwargs):
        super().export(*args, model_format=model_format, **kwargs)

    @staticmethod
    def _convert_replacement(model, pattern, *args, remove_qconfig=True, **kwargs):
        model = torch.fx.symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model

        # for qdq model
        # replacement_entries_qdq = [
        #    ([torch.ao.nn.intrinsic.modules.fused.ConvReLU2d,edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize], model_quant_utils.OffsetScaleShift.from_conv_relu_fq),
        #    ([edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize, torch.nn.BatchNorm2d, edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize], model_quant_utils.OffsetScaleShift.from_fq_bn_fq),
        #    ([edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize], model_quant_utils.OffsetScaleShift.from_fq),
        # }

        # for converted model
        replacement_entries_converted = [
            ([torch.quantize_per_tensor], quant_utils.OffsetScaleShift.from_q),
            ([torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d], quant_utils.OffsetScaleShift.from_qbn),
            ([torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d],
             quant_utils.OffsetScaleShift.from_qconvrelu),
            (['dequantize'], quant_utils.OffsetScaleShift.from_dq)
        ]

        for replacement_pattern, replacement_function in replacement_entries_converted:
            matches = edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer.straight_type_chain_searcher(model,
                                                                                                      replacement_pattern)
            for no_of_module_replaced, (start, end) in enumerate(matches):
                new_fq_module = replacement_function(model, start, end)
                edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer._replace_pattern(model, start, end, new_fq_module,
                                                                                    no_of_module_replaced)
            #
        #
        return model
