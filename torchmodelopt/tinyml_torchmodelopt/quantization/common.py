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

import torch
from edgeai_torchmodelopt.xmodelopt.quantization.v2 import ModelQuantFormat
import edgeai_torchmodelopt


class TinyMLQuantizationVersion():
    NO_QUANTIZATION = 0
    QUANTIZATION_GENERIC = 1
    QUANTIZATION_TINIE = 2

    @classmethod
    def get_dict(cls):
        return {k:v for k,v in cls.__dict__.items() if not k.startswith("__")}

    @classmethod
    def get_choices(cls):
        return {v:k for k,v in cls.__dict__.items() if not k.startswith("__")}


class TinyMLModelQuantFormat(ModelQuantFormat):
    TINIE_INT_MODEL = "TINIE_INT_MODEL"
    _NUM_FORMATS_ = ModelQuantFormat._NUM_FORMATS_ + 1


class GenericTinyMLQATFxModuleBase(edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule):
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
    
    