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
        quant_error_min = diff_output.abs().min().item()
        quant_error_max = diff_output.abs().max().item()
        quant_error_mean = diff_output.abs().mean().item()
        quant_snr_db = (torch.log10((float_output**2).mean() / (diff_output**2).mean()) * 10).item()
        quant_psnr_db = (torch.log10((float_output**2).max() / (diff_output**2).mean()) * 10).item()        
        quant_absmu_by_sigma = (float_output.abs().mean() / diff_output.std()).item()
        diff_output_stats = dict(snr_db=quant_snr_db, psnr_db=quant_psnr_db, absmu_by_sigma=quant_absmu_by_sigma, 
                                 min=quant_error_min, max=quant_error_max, mean=quant_error_mean)
        return diff_output_stats
    
    