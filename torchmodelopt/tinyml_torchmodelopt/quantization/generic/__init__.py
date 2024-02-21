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
import os.path

import torch
import edgeai_torchmodelopt
from ..common import TinyMLQuantizationVersion, TinyMLModelQuantFormat


class GenericTinyMLQATFxModule(edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule):
    def __init__(self, model, *args, qconfig_type=None,  **kwargs):
        qconfig_type_in = qconfig_type
        qconfig_type_default_dict = dict(weight=dict(bitwidth=8, qscheme=torch.per_channel_symmetric, power2_scale=True),
                                         activation=dict(bitwidth=8, qscheme=torch.per_tensor_symmetric,
                                         power2_scale=True, range_max=None, fixed_range=False))
        if qconfig_type_in is not None and isinstance(qconfig_type_in, dict):
            qconfig_type = copy.deepcopy(qconfig_type_default_dict)
            qconfig_type.update(qconfig_type_in)
        elif isinstance(qconfig_type_in, str):
            qconfig_type = qconfig_type_in
        else:
            qconfig_type = copy.deepcopy(qconfig_type_default_dict)
        #
        super().__init__(model, *args, qconfig_type=qconfig_type, **kwargs)

    def convert(self, *args, model_quant_format=TinyMLModelQuantFormat.INT_MODEL, **kwargs):
        return super().convert(*args, model_quant_format=model_quant_format, **kwargs)

    def export(self, *args, model_quant_format=TinyMLModelQuantFormat.INT_MODEL, **kwargs):
        super().export(*args, model_quant_format=model_quant_format, **kwargs)
