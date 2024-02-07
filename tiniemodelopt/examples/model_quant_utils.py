import edgeai_torchmodelopt.xmodelopt.quantization.v2.quant_fx_base
import torch


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

    # x_hat = scale *  torch.pow(torch.tensor([2.0]), -shift)
    # mse = torch.mean((x-x_hat)**2)
    # if print_mse:
    #     print(mse)
    return shift, scale


class OffsetScaleShift(torch.nn.Module):
    def __init__(self, offset, scale, shift, c_n, c_p, quantize_per_channel=False, use_floor=True):
        super().__init__()
        self.c_n = c_n
        self.c_p = c_p
        self.quantize_per_channel = quantize_per_channel
        self.use_floor = use_floor
        self.register_buffer('offset', offset)
        self.register_buffer('scale', scale)
        self.register_buffer('shift', shift)

    def extra_repr(self):
        return 'num_bits_offset={num_bits_offset}, num_bits_scale={num_bits_scale}, num_bits_shift={num_bits_shift}, clip={c_p}'.format(**self.__dict__)

    def forward(self, x):
        if self.quantize_per_channel:
            y = (x + self.offset.unsqueeze(0).unsqueeze(2).unsqueeze(3))*self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            y = y * self.shift.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            y = (x + self.offset)*self.scale
            y = y * self.shift
        #
        if self.use_floor:
            y = torch.floor(y).clamp(min=self.c_n, max=self.c_p) #the floor operation mimics the actual shift and bit select in hardware
        else:
            y = torch.round(y).clamp(min=self.c_n,max=self.c_p)
        return y

    @staticmethod
    def from_fq_bn_fq(fq_module1, bn_module, fq_module2):
        fq_observed1 = fq_module1.activation_post_process
        scale, zero_point = fq_observed1.calculate_qparams()
        fq_observed2 = fq_module1.activation_post_process
        scale, zero_point = fq_observed2.calculate_qparams()
        return torch.nn.Identity()

    @staticmethod
    def from_cbn_fq(cbn_module, fq_module):
        fq_observed = fq_module.activation_post_process
        scale, zero_point = fq_observed.calculate_qparams()
        return torch.nn.Identity()

