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

# Imports Torch
import torch
import torch.ao.quantization
from torch.ao.quantization import QConfig, QConfigMapping

from .observer_types import *


def get_symmetric_power2_qconfig(weight_qscheme=torch.per_channel_symmetric, activation_qscheme=torch.per_tensor_symmetric, power2_scale=True):
    weight_fake_quant = torch.ao.quantization.fake_quantize.FakeQuantize.with_args(observer=SimplePerChannelWeightObserver, quant_min=-128, quant_max=127, qscheme=weight_qscheme, dtype=torch.qint8, power2_scale=power2_scale)
    activation_fake_quant = torch.ao.quantization.fake_quantize.FakeQuantize.with_args(observer=SimpleActivationObserver, quant_min=0, quant_max=255, qscheme=activation_qscheme, dtype=torch.quint8, power2_scale=power2_scale)
    qconfig = QConfig(weight=weight_fake_quant, activation=activation_fake_quant)
    return qconfig


def get_default_qconfig(weight_qscheme=torch.per_channel_symmetric, activation_qscheme=torch.per_tensor_symmetric, power2_scale=True):
    return get_symmetric_power2_qconfig(weight_qscheme, activation_qscheme, power2_scale)


def get_default_qconfig_mapping(qconfig=None):
    if qconfig is None:
        qconfig = get_default_qconfig()
    #
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    return qconfig_mapping
    