import numpy as np
import torch
from torch.ao.quantization import QuantType
from torch.ao.quantization.quantize_fx import ConvertCustomConfig

def compute_shift_scale(x, num_bits_shift=8, num_bits_scale=8, print_mse =True):
    x_abs = x.abs()
    x_sign = x.sign()
    u_p = 2**(num_bits_scale)-1
    power_of_2 = torch.floor(torch.log2(u_p/x_abs))*torch.tensor([1.0])
    #print('power_of_2',power_of_2)
    shift = power_of_2.clamp(min=0, max=(2**(num_bits_shift)-1))
    #print('shift',shift)
    scale = x_abs * torch.pow(torch.tensor([2.0]), shift)
    #print('scale',scale)

    #nan fix:
    mask = torch.isnan(scale)
    #print(mask)
    scale[mask] = 0
    shift[mask] = 1

    assert(torch.sum(scale > u_p) == 0)

    scale = x_sign*torch.round(scale)

    x_hat = scale *  torch.pow(torch.tensor([2.0]), -shift)
    mse = torch.mean((x-x_hat)**2)
    # if print_mse:
    #     print(mse)
    return shift, scale


class CustomQuantizedBatchNorm2d(torch.nn.Module):
    def __init__(self, add_value, mul_value, shift_value):
        super().__init__()
        self.add_value = add_value
        self.mul_vaue = mul_value
        self.shift_value = shift_value

    def forward(self, x):
        shift_to_mul = np.power(2, self.shift_value)
        return ((x + self.add_value) * self.mul_vaue) * shift_to_mul

    @staticmethod
    def from_observed(observed):
        if hasattr(observed, 'with_convert_custom_config') and observed.with_convert_custom_config:
            sigma = torch.sqrt(observed.running_var + observed.eps)
            fused_add_value = observed.running_mean / sigma + observed.bias
            fused_mul_value = observed.weight / sigma
            shift, scale = compute_shift_scale(fused_mul_value)
            return CustomQuantizedBatchNorm2d(fused_add_value.detach().numpy(), shift.detach().numpy(), scale.detach().numpy())
        else:
            return observed


def get_convert_custom_config(replace_input_norm=True):
    convert_custom_config = ConvertCustomConfig()
    if replace_input_norm:
        convert_custom_config.set_observed_to_quantized_mapping(torch.nn.BatchNorm2d, CustomQuantizedBatchNorm2d, QuantType.STATIC)
    #
    return convert_custom_config
