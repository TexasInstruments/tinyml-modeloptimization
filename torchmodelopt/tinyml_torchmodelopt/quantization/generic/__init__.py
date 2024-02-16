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


class GenericTinyMLQATFxModule(edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule):
    def __init__(self, model, qconfig_type=None, total_epochs=None):
        qconfig_type_default = dict(weight=dict(bitwidth=8, qscheme=torch.per_channel_symmetric, power2_scale=True),
                                    activation=dict(bitwidth=8, qscheme=torch.per_tensor_symmetric,
                                    power2_scale=True, range_max=None, fixed_range=False))
        if qconfig_type is not None:
            qconfig_type_in = qconfig_type
            qconfig_type = copy.deepcopy(qconfig_type_default)
            qconfig_type.update(qconfig_type_in)
        else:
            qconfig_type = copy.deepcopy(qconfig_type_default)
        #
        super().__init__(model, qconfig_type, total_epochs)

    def convert(self, inplace=False, device='cpu', convert_custom_config=None, backend_config=None):
        # TODO: remove thse lines - not needed
        # native pytorch int8 quantization
        # convert_custom_config = model_export_torch_utils.get_convert_custom_config()
        # backend_config = torch.ao.quantization.backend_config.get_native_backend_config()
        model = self if inplace else copy.deepcopy(self)
        model.module = super().convert(inplace=inplace, device=device, convert_custom_config=convert_custom_config, backend_config=backend_config).module

    def export(self, *args, **kwargs):
        super().export(*args, **kwargs)
