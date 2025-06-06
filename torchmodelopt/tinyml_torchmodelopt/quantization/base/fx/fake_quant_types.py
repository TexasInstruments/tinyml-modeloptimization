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

import warnings
import copy
import torch
import torch.ao.quantization


class SoftFakeQuantize(torch.ao.quantization.fake_quantize.FakeQuantize):
    """
    A custom quantization class that extends FakeQuantize.
    This class can be used to define specific quantization behavior.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Temperature parameter for sigmoid approximation
        self.temperature = 6.0

    def update_temperature(self, t):
        self.temperature = t

    def soft_round(self, x):
        '''
        Smooth quantization using temperature-controlled approximation of round
        '''
        delta = x - self.floor_ste(x)
        soft_delta = torch.sigmoid(self.temperature * (delta - 0.5))
        y = self.floor_ste(x) + soft_delta
        return y

    def hard_round(self, x):
        y = torch.round(x)
        return y

    # def propagate_quant_ste(self, x, y):
    #     # this works functionally as STE, but exports an onnx graph containing
    #     # all the operators used to compute y as well
    #     # out = x + (y - x).detach()
    #     #
    #     # this is another way of doing STE. in this case the operators used to generate y are skipped from onnx graph
    #     out = x.clone()
    #     out.data = y.data
    #     return out
    #
    # def floor_ste(self, x):
    #     return self.propagate_quant_ste(x, torch.floor(x))

    def floor_ste(self, x):
        return x + (torch.floor(x)-x).detach()

    def forward(self, X: torch.Tensor):
        Y = super().forward(X)

        dataflat = X.reshape(X.shape[0], -1)
        data_normalized = (dataflat / self.scale.unsqueeze(-1)) + self.zero_point.unsqueeze(-1)

        if self.training:
            # Apply smooth quantization
            data_soft_quant = self.soft_round(data_normalized)
            data_soft_clamped = torch.clamp(data_soft_quant, min=self.quant_min, max=self.quant_max)
            data_soft_dequant = (data_soft_clamped - self.zero_point.unsqueeze(-1)) * self.scale.unsqueeze(-1)
            qdata_soft = data_soft_dequant.reshape(X.shape)
            output = Y + qdata_soft - Y
        else:
            # data_hard_quant = self.hard_round(data_normalized)
            # data_hard_clamped = torch.clamp(data_hard_quant, min=self.quant_min, max=self.quant_max)
            # data_hard_dequant = (data_hard_clamped - self.zero_point.unsqueeze(-1)) * self.scale.unsqueeze(-1)
            # qdata_hard = data_hard_dequant.reshape(X.shape)
            # output = qdata_hard
            output = Y

        return output