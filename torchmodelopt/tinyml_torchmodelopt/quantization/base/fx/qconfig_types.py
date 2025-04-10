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
from torch.ao.quantization import QConfig, QConfigMapping

from . import observer_types

def get_8bit_layers():
    layers = []
    # layers += ['conv1', 'bn1', 'relu1']

    # layers += ['pointwise2', 'bn22', 'relu22']
    # layers += ['pointwise3', 'bn32', 'relu32']
    # layers += ['pointwise4', 'bn42', 'relu42']
    # layers += ['pointwise5', 'bn52', 'relu52']

    # layers += ['pointwise2', 'bn21', 'relu21']
    # layers += ['depthwise3', 'bn31', 'relu31']
    # layers += ['depthwise4', 'bn41', 'relu41']
    # layers += ['depthwise5', 'bn51', 'relu51']

    # layers += ['avgpool', 'flatten6']
    # layers += ['fc6']
    return layers

def get_4bit_layers():
    layers = []

    layers += ['conv1', 'bn1', 'relu1']

    layers += ['pointwise2', 'bn22', 'relu22']
    # layers += ['pointwise3', 'bn32', 'relu32'] # WORST in 4bit
    # layers += ['pointwise4', 'bn42', 'relu42']
    layers += ['pointwise5', 'bn52', 'relu52']

    layers += ['depthwise2', 'bn21', 'relu21']
    layers += ['depthwise3', 'bn31', 'relu31']
    layers += ['depthwise4', 'bn41', 'relu41']
    layers += ['depthwise5', 'bn51', 'relu51']

    layers += ['avgpool', 'flatten6']
    layers += ['fc6']
    return layers

def get_default_qconfig(qconfig_dict=None):
    '''
    This default qconfig uses symmetric, power2 quantization.
    It can be changed by passing appropriate qconfig_dict.
    '''
    qconfig_dict = qconfig_dict or dict()
    weight_qconfig = qconfig_dict.get('weight', dict())
    weight_dtype = weight_qconfig.get('dtype', torch.qint8)
    weight_bitwidth = weight_qconfig.get('bitwidth', 8)
    weight_quant_min = weight_qconfig.get('quant_min', -(2 ** (weight_bitwidth - 1)))
    weight_quant_max = weight_qconfig.get('quant_max', (2 ** (weight_bitwidth - 1)) - 1)
    weight_qscheme = weight_qconfig.get('qscheme', torch.per_channel_symmetric)
    weight_power2_scale = weight_qconfig.get('power2_scale', True)
    weight_range_max = weight_qconfig.get('range_max', None)
    weight_fixed_range = weight_qconfig.get('fixed_range', False)
    weight_histogram_range = weight_qconfig.get('histogram_range', False)

    activation_qconfig = qconfig_dict.get('activation', dict())
    activation_dtype = activation_qconfig.get('dtype', torch.quint8)
    activation_bitwidth = activation_qconfig.get('bitwidth', 8)
    activation_quant_min = activation_qconfig.get('quant_min', 0)
    activation_quant_max = activation_qconfig.get('quant_max', (2 ** activation_bitwidth) - 1)
    activation_qscheme = activation_qconfig.get('qscheme', torch.per_tensor_symmetric)
    activation_power2_scale = activation_qconfig.get('power2_scale', True)
    activation_range_max = activation_qconfig.get('range_max', None)
    activation_fixed_range = activation_qconfig.get('fixed_range', False)
    activation_histogram_range = activation_qconfig.get('histogram_range', False)
    bias_calibration_factor = activation_qconfig.get('bias_calibration_factor', 0.0)

    if weight_qscheme == torch.per_channel_symmetric:
        # we don't have a historgram observer tht can do per_channel_symmetric - so use minmax
        weight_observer_base_class = torch.ao.quantization.PerChannelMinMaxObserver
    elif weight_histogram_range:
        weight_observer_base_class = observer_types.MovingAverageRangeShrinkFastHistogramObserver \
                    if weight_histogram_range == 1 else torch.ao.quantization.HistogramObserver
    else:
        weight_observer_base_class = torch.ao.quantization.MinMaxObserver
    #
    weight_fake_quant = torch.ao.quantization.fake_quantize.FakeQuantize.with_args(
        observer=observer_types.get_weight_observer_type(base_class=weight_observer_base_class),
        quant_min=weight_quant_min, quant_max=weight_quant_max,
        qscheme=weight_qscheme, dtype=weight_dtype, power2_scale=weight_power2_scale,
        range_max=weight_range_max, fixed_range=weight_fixed_range)

    if activation_histogram_range:
        activation_observer_base_class = observer_types.MovingAverageRangeShrinkFastHistogramObserver \
            if activation_histogram_range==1 else torch.ao.quantization.HistogramObserver
    else:
        activation_observer_base_class = torch.ao.quantization.MovingAverageMinMaxObserver
    #
    activation_fake_quant = torch.ao.quantization.fake_quantize.FakeQuantize.with_args(
        observer=observer_types.get_activation_observer_type(base_class=activation_observer_base_class),
        quant_min=activation_quant_min, quant_max=activation_quant_max,
        qscheme=activation_qscheme, dtype=activation_dtype, power2_scale=activation_power2_scale,
        range_max=activation_range_max, fixed_range=activation_fixed_range,
        bias_calibration_factor=bias_calibration_factor)

    qconfig = QConfig(weight=weight_fake_quant, activation=activation_fake_quant)
    return qconfig


def get_default_qconfig_mapping(qconfig_type=None):
    qconfig_dict = qconfig_type
    qconfig = {}
    mixed_precision = qconfig_dict.get('mixed_precision', [])
    if isinstance(qconfig_dict, dict) or qconfig_dict is None:
        qconfig_type = get_default_qconfig(qconfig_dict=qconfig_dict)
        if 8 in mixed_precision:
            qconfig_dict['weight']['bitwidth'] = 8
            qconfig[8] = dict()
            qconfig[8]['qconfig'] = get_default_qconfig(qconfig_dict=qconfig_dict)
            qconfig[8]['layers'] = get_8bit_layers()
        if 4 in mixed_precision:
            qconfig_dict['weight']['bitwidth'] = 4
            qconfig[4] = dict()
            qconfig[4]['qconfig'] = get_default_qconfig(qconfig_dict=qconfig_dict)
            qconfig[4]['layers'] = get_4bit_layers()
        if 2 in mixed_precision:
            qconfig_dict['weight']['bitwidth'] = 2
            qconfig[2] = dict()
            qconfig[2]['qconfig'] = get_default_qconfig(qconfig_dict=qconfig_dict)
            qconfig[2]['layers'] = get_4bit_layers()
    #
    if not isinstance(qconfig_type, QConfig):
        raise RuntimeError("Unrecognized type of qconfig_type")
    #
    qconfig_mapping = QConfigMapping().set_global(qconfig_type)
    for bitwidth in mixed_precision:
        for layer in qconfig[bitwidth]['layers']:
            qconfig_mapping.set_module_name(layer, qconfig[bitwidth]['qconfig'])
    print(qconfig_mapping)
    return qconfig_mapping
    