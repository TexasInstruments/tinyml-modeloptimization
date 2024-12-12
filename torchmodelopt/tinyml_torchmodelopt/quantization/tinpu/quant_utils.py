import torch
from torch import fx
from torch.fx import GraphModule

def compute_offset_scale_shift(offset, weight, num_bits_shift=5, num_bits_scale=1, print_mse=False):
    """
    Represent offset, weight using add, mult and right shift
    :param offset: additive offset
    :param weight: multiplicative weight
    :param num_bits_shift: number of bits to represent the shift value. this is not the number of bits to shift (which depends on the weight value), but the number of bits to represent the shift value.
    :param num_bits_scale: number of bits to represent the scale value.
    :param print_mse:
    :return:
    """
    weight_abs = weight.abs()
    weight_sign = weight.sign()
    scale_max = (2**num_bits_scale)-1
    power_of_2 = torch.floor(torch.log2(scale_max/weight_abs))*torch.tensor([1.0])
    shift = power_of_2.clamp(min=0, max=((2**num_bits_shift)-1))
    scale = weight_abs * torch.pow(torch.tensor([2.0]), shift)

    mask = torch.isnan(scale)
    scale[mask] = 0
    shift[mask] = 1

    if torch.sum(scale > scale_max) != 0:
        raise RuntimeError(
            f"Error in Quant convert:compute_offset_scale_shift. Output multipliation could not be converted. \n"
            f"Invalid in output multipliation value: {weight.cpu().numpy()} \n"
            f"Make sure that the model is trained properly with good hyper parameters. "
            f"(try adjusting: training epochs, learning rate, QAT after float training etc): \n"
        )
    #

    scale = weight_sign*torch.round(scale)
    shift_mult = torch.pow(torch.tensor([2.0]), -shift)

    if print_mse:
        weight_hat = scale * torch.pow(torch.tensor([2.0]), -shift)
        mse = torch.mean((weight-weight_hat)**2)
        print(mse)
    #

    # add round offset to the offset. since the offset is before the scale, divide it by scale before adding
    shift_round_offset = torch.pow(torch.tensor([2.0]), (shift-1)) / scale
    offset = torch.round(offset + shift_round_offset)

    return offset, scale, shift_mult

def propagate_quant_args_after_flatten(main_module : GraphModule) -> GraphModule:
    """
    Propagates scale and zero point arguments into a torch.quantize_per_tensor
    after a Flatten layer. Simplifies the graph to assist TI-NPU quant conversion
    pattern-matching (_convert_replacement in quant_fx.py).
    :main_module: input GraphModule
    :return: the transformed GraphModule
    """
    main_interpret = torch.fx.Interpreter(main_module)
    main_module_nodes:list[fx.Node] = list(main_module.graph.nodes)
    modules_in_main_graph = dict(main_module.named_modules())
    main_module_node_num = len(main_module_nodes)
    main_index = 0
    while main_index < main_module_node_num:
        main_node = main_module_nodes[main_index]
        if (main_node.op == 'call_module' and \
            isinstance(modules_in_main_graph[main_node.target],torch.nn.Flatten)):
            if main_node.next.op == 'get_attr' and 'scale_0' in main_node.next.target and \
               main_node.next.next.op == 'get_attr' and \
               'zero_point_0' in main_node.next.next.target and \
               main_node.next.next.next.op == 'call_function' and \
               main_node.next.next.next.target == torch.quantize_per_tensor:
                scale_node = main_node.next
                zerop_node = main_node.next.next
                quant_node = zerop_node.next
                scale = main_interpret.run_node(scale_node)
                zerop = main_interpret.run_node(zerop_node)
                quant_node.update_arg(1, scale)
                quant_node.update_arg(2, zerop)
                main_module.graph.erase_node(scale_node)
                main_module.graph.erase_node(zerop_node)
        main_index +=1
    main_module.graph.lint()
    main_module.recompile()
    return main_module

def swap_flatten_quant(main_module : GraphModule) -> GraphModule:
    """
    Rewrite (Flatten, torch.quantize_per_tensor) as (torch.quantize_per_tensor, Flatten) so that
    the input to the Flatten layer is quantized. This arrangement is more optimal for model
    compilation and inference.
    :main_module: input GraphModule
    :returns: the transformed GraphModule
    precondition: propagate_quant_args_after_flatten
    """
    main_module_nodes:list[fx.Node] = list(main_module.graph.nodes)
    modules_in_main_graph = dict(main_module.named_modules())
    main_module_node_num = len(main_module_nodes)
    main_index = 0
    while main_index < main_module_node_num:
        main_node = main_module_nodes[main_index]
        if main_node.op == 'call_module' and \
            isinstance(modules_in_main_graph[main_node.target],torch.nn.Flatten):
            assert main_index > 0 , "Internal ERROR: torch.nn.Flatten is input to graph"
            use_keys = list(main_node.users.keys())
            if len(use_keys) == 1 and torch.quantize_per_tensor == use_keys[0].target:
                flatten_input_node = main_module_nodes[main_index-1]
                orig_quant_node = use_keys[0]
                with main_module.graph.inserting_after(flatten_input_node):
                    new_quant_node = main_module.graph.call_function(torch.quantize_per_tensor,\
                                            args=(main_node.args[0],) + orig_quant_node.args[1:],\
                                                    kwargs=orig_quant_node.kwargs)
                    main_node.replace_input_with(old_input=flatten_input_node,\
                                                 new_input=new_quant_node)
                    orig_quant_node.replace_all_uses_with(main_node)
                main_module.graph.erase_node(orig_quant_node)
                assert flatten_input_node.next == new_quant_node, \
                    "Internal ERROR: incorrect swap_flatten_quant transformation"
        main_index +=1
    main_module.graph.lint()
    main_module.recompile()
    return main_module

class MultiplyModule(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.mul(x, self.value)


class TINPUOffsetScaleShift(torch.nn.Module):
    def __init__(self, offset, mult, shift_mult, quant_min, quant_max, quantize_per_channel=False, use_floor=True, ndim=4, dim=1):
        super().__init__()
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.quantize_per_channel = quantize_per_channel
        self.use_floor = use_floor
        if ndim == 4 and dim == 1:
            self.register_buffer('offset', offset.reshape(1,-1,1,1))
            self.register_buffer('mult', mult.reshape(1,-1,1,1))
            self.register_buffer('shift_mult', shift_mult.reshape(1,-1,1,1))
        elif ndim == 2 and dim == 1:
            self.register_buffer('offset', offset.reshape(1,-1))
            self.register_buffer('mult', mult.reshape(1,-1))
            self.register_buffer('shift_mult', shift_mult.reshape(1,-1))
        elif ndim == 1:
            self.register_buffer('offset', offset.reshape(-1))
            self.register_buffer('mult', mult.reshape(-1))
            self.register_buffer('shift_mult', shift_mult.reshape(-1))
        else:
            raise RuntimeError('Invalid dimensions')
        #

    def extra_repr(self):
        return f'offset={self.offset}, mult={self.mult}, shift={self.shift}, quant_min={self.quant_min}, quant_max={self.quant_max}'

    def forward(self, x):
        y = (x + self.offset) * self.mult
        y = y * self.shift_mult
        if self.use_floor:
            y = torch.floor(y).clamp(min=self.quant_min, max=self.quant_max) #the floor operation mimics the actual shift and bit select in hardware
        else:
            y = torch.round(y).clamp(min=self.quant_min,max=self.quant_max)
        return y


class TINPUQuantizedReplacement:
    @staticmethod
    def from_child_module(model, start, end):
        '''
        copy quant scale, zero_point from child module
        '''
        named_modules = dict(model.named_modules())
        module = named_modules[start.target]
        if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
            return module

        named_children = list(module.named_children())
        if len(named_children) > 0:
            last_child_name, last_child = named_children[-1]
            if hasattr(last_child, 'scale') and hasattr(last_child, 'zero_point'):
                module.scale = last_child.scale
                module.zero_point = last_child.zero_point
                return module

        return module

    @staticmethod
    def _get_scale_zero_point_attrs_from_model(model, node):
        scale_attr_name = node.name.replace('.', '_') + '_scale_0'
        zero_point_attr_name = node.name.replace('.', '_') + '_zero_point_0'
        if hasattr(model, scale_attr_name) and hasattr(model, zero_point_attr_name):
            scale = getattr(model, scale_attr_name)
            zero_point = getattr(model, zero_point_attr_name)
        else:
            scale = zero_point = None
        #
        return scale, zero_point

    @staticmethod
    def _get_scale_zero_point_from_previous(model, node):
        named_modules = dict(model.named_modules())
        prev_node = node.prev
        prev_module_found = False
        scale = zero_point = None
        while not prev_module_found and prev_node:
            if prev_node.target in named_modules:
                previous_module = named_modules[prev_node.target]
                if hasattr(previous_module, 'scale') and hasattr(previous_module, 'zero_point'):
                    scale = previous_module.scale
                    zero_point = previous_module.zero_point
                #
                return scale, zero_point
            else:
                scale, zero_point = __class__._get_scale_zero_point_attrs_from_model(model, prev_node)
                if scale is not None and zero_point is not None:
                    return scale, zero_point
                #
            #
            prev_node = prev_node.prev
        #
        return scale, zero_point

    @staticmethod
    def from_q(model, start, end):
        q_node = start
        scale = q_node.args[1]
        zero_point = q_node.args[2]
        if hasattr(q_node.args[1], "target") and hasattr(q_node.args[2], "target"):
            scale = getattr(model, q_node.args[1].target)
            zero_point = getattr(model, q_node.args[2].target)
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(zero_point*0.0, 1/scale, num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=4, dim=1)
        oss_module.scale = scale
        oss_module.zero_point = zero_point
        return oss_module

    @staticmethod
    def from_q_id(model, start, end):
        return __class__.from_q(model, start, end)

    @staticmethod
    def from_q_qbn(model, start, end):
        qbn_module = dict(model.named_modules())[end.target]
        bn_sigma = torch.sqrt(qbn_module.running_var + qbn_module.eps)

        scale2 = qbn_module.scale
        zero_point2 = qbn_module.zero_point

        oss_offset = (- qbn_module.running_mean + qbn_module.bias*bn_sigma)
        # first get the effective weight due to batchnorm
        combined_weight = (qbn_module.weight / bn_sigma)
        # then modify the weight by output scale so that the output is converted to output scale
        combined_weight = combined_weight / scale2
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(oss_offset, combined_weight, num_bits_scale=8)

        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=4, dim=1)
        oss_module.scale = scale2
        oss_module.zero_point = zero_point2
        return oss_module

    @staticmethod
    def from_flatten_with_q(model, start, end):
        """
        Set scale and zero point of a Flatten module.
        dequantize, torch.quantize_per_tensor, torch.nn.Flatten
        -> torch.nn.Flatten
        """
        module = dict(model.named_modules())[end.target]
        q_node =start.next
        scale = q_node.args[1]
        zero_point = q_node.args[2]
        module.scale = scale
        module.zero_point = zero_point
        return module

    @staticmethod
    def from_module_with_dq(model, start, end):
        module = dict(model.named_modules())[start.target]
        # oss_module = __class__.from_dq(model, start.next, end)
        # seq_module = torch.nn.Sequential(module, oss_module)
        # seq_module.scale = oss_module.scale
        # seq_module.zero_point = oss_module.zero_point
        scale, zero_point = __class__._get_scale_zero_point_from_previous(model, start.next)
        mult_module = MultiplyModule(scale)
        mult_module.scale = 1.0
        mult_module.zero_point = 0.0
        seq_module = torch.nn.Sequential(module, mult_module)
        seq_module.scale = mult_module.scale
        seq_module.zero_point = mult_module.zero_point
        return seq_module

    @staticmethod
    def from_module_with_q(model, start, end):
        module = dict(model.named_modules())[start.target]
        oss_module = __class__.from_q(model, start.next, end)
        seq_module = torch.nn.Sequential(module, oss_module)
        seq_module.scale = oss_module.scale
        seq_module.zero_point = oss_module.zero_point
        return seq_module

    @staticmethod
    def from_qconv_relu(model, start, end, with_relu=True):
        # zero_point_offset_for_activation = -128
        named_modules = dict(model.named_modules())

        qconvrelu_module = named_modules[start.target]
        conv_module = torch.nn.Conv2d(qconvrelu_module.in_channels, qconvrelu_module.out_channels,
                                      kernel_size=qconvrelu_module.kernel_size, stride=qconvrelu_module.stride,
                                      padding=qconvrelu_module.padding, dilation=qconvrelu_module.dilation,
                                      groups=qconvrelu_module.groups, bias=False)

        weight = qconvrelu_module.weight()
        per_channel = (weight.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine))

        qweight = weight.data.detach().int_repr()
        conv_module.weight.data.copy_(qweight)

        weight_scale = weight.q_per_channel_scales() if per_channel else weight.q_scale()
        weight_zero_point = weight.q_per_channel_zero_points() if per_channel else weight.q_zero_point()
        input_scale = named_modules[start.prev.target].scale
        # input_zero_point = named_modules[start.prev.target].zero_point

        acc_scale = weight_scale * input_scale
        bias_scale = acc_scale
        bias_zero_point = weight_zero_point
        bias = qconvrelu_module.bias()

        # qbias = (torch.round(bias / bias_scale) + bias_zero_point).float()
        if per_channel:
            qbias = torch.quantize_per_channel(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        else:
            qbias = torch.quantize_per_tensor(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        #
        qbias = qbias.int_repr()

        # conv_module.bias.data.copy_(qbias)
        relative_mult = (acc_scale / qconvrelu_module.scale).float()
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(qbias, relative_mult)
        if with_relu:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -255, 255)
            seq_module = torch.nn.Sequential(conv_module, oss_module, torch.nn.ReLU(), torch.nn.Hardtanh(0, 255))
        else:
            # this clip is left to -255, 255 here, assuming that there is an Add and ReLU after this.
            # Otherwise it should be -128, 127
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -255, 255)
            seq_module = torch.nn.Sequential(conv_module, oss_module)
        #
        seq_module.scale = qconvrelu_module.scale
        seq_module.zero_point = qconvrelu_module.zero_point
        return seq_module

    @staticmethod
    def from_qlinear(model, start, end, with_relu=False):
        # zero_point_offset_for_activation = -128

        # named_modules = dict(model.named_modules())

        qlinear_module = dict(model.named_modules())[start.target]
        linear_module = torch.nn.Linear(qlinear_module.in_features, qlinear_module.out_features, bias=False)

        weight = qlinear_module.weight()
        per_channel = (weight.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine))

        qweight = weight.data.detach().int_repr()
        linear_module.weight.data.copy_(qweight)

        weight_scale = weight.q_per_channel_scales() if per_channel else weight.q_scale()
        weight_zero_point = weight.q_per_channel_zero_points() if per_channel else weight.q_zero_point()

        # if the previous node is a module, use scale from there.
        # otherwise we have to go back further.
        # if start.prev.target in named_modules and hasattr(named_modules[start.prev.target], 'scale'):
        #     input_scale = named_modules[start.prev.target].scale
        #     input_zero_point = named_modules[start.prev.target].zero_point
        # else:
        #     prev_node = start.prev
        #     prev_module_found = False
        #     while not prev_module_found and prev_node:
        #         if prev_node.target in named_modules and hasattr(named_modules[prev_node.target], 'scale'):
        #             input_scale = named_modules[prev_node.target].scale
        #             input_zero_point = named_modules[prev_node.target].zero_point
        #             prev_module_found = True
        #         #
        #         if not prev_module_found:
        #             prev_node = prev_node.prev
        #         #
        #     #
        # #
        input_scale, input_zero_point = __class__._get_scale_zero_point_from_previous(model, start)

        acc_scale = weight_scale * input_scale
        bias_scale = acc_scale
        bias_zero_point = weight_zero_point
        bias = qlinear_module.bias()

        # qbias = (torch.round(bias / bias_scale) + bias_zero_point).float()
        if per_channel:
            qbias = torch.quantize_per_channel(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        else:
            qbias = torch.quantize_per_tensor(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        #
        qbias = qbias.int_repr()

        # conv_module.bias.data.copy_(qbias)
        relative_mult = (acc_scale / qlinear_module.scale).float()
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(qbias, relative_mult)

        if with_relu:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255, ndim=2, dim=1)
            seq_module = torch.nn.Sequential(linear_module, oss_module, torch.nn.ReLU(), torch.nn.Hardtanh(0, 255))
        else:
            oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=2, dim=1)
            seq_module = torch.nn.Sequential(linear_module, oss_module)
        #
        seq_module.scale = qlinear_module.scale
        seq_module.zero_point = qlinear_module.zero_point
        return seq_module

    @staticmethod
    def from_qlinear_relu(model, start, end, with_relu=True):
        return __class__.from_qlinear(model, start, end, with_relu=with_relu)

    @staticmethod
    def from_passthrough_module(model, start, end):
        named_modules = dict(model.named_modules())
        passthrough_module = named_modules[start.target]
        if hasattr(passthrough_module, 'scale') and hasattr(passthrough_module, 'zero_point'):
            return passthrough_module

        scale, zero_point = __class__._get_scale_zero_point_attrs_from_model(model, start)
        if scale is not None and zero_point is not None:
            passthrough_module.scale = scale
            passthrough_module.zero_point = zero_point
            return passthrough_module

        if start.prev.target in named_modules:
            prev_module = named_modules[start.prev.target]
            passthrough_module.scale = prev_module.scale
            passthrough_module.zero_point = prev_module.zero_point
            return passthrough_module

        scale, zero_point = __class__._get_scale_zero_point_attrs_from_model(model, start.prev)
        if scale is not None and zero_point is not None:
            passthrough_module.scale = scale
            passthrough_module.zero_point = zero_point
            return passthrough_module

        return passthrough_module

    @staticmethod
    def from_dq(model, start, end):
        id_module = torch.nn.Identity()
        id_module.scale = 1.0 #prev_module.scale
        id_module.zero_point = 0.0
        return id_module

    @staticmethod
    def from_dq_with_dq(model, start, end):
        # named_modules = dict(model.named_modules())
        scale, zero_point = __class__._get_scale_zero_point_from_previous(model, start)
        id_module = torch.nn.Identity()
        id_module.scale = scale
        id_module.zero_point = 0.0
        mult_module = MultiplyModule(id_module.scale)
        mult_module.scale = 1.0
        mult_module.zero_point = 0.0
        output_module = torch.nn.Sequential(id_module, mult_module)
        output_module.scale = mult_module.scale
        output_module.zero_point = mult_module.zero_point
        return output_module

    @staticmethod
    def from_avgpool2d(model, start, end):
        # named_modules = dict(model.named_modules())
        pool_module = dict(model.named_modules())[start.target]
        scale, zero_point = __class__._get_scale_zero_point_from_previous(model, start)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
            zero_point = torch.tensor(zero_point)

        total_kernel_area = pool_module.kernel_size[0]*pool_module.kernel_size[1]
        mult_module = MultiplyModule(total_kernel_area)

        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(torch.tensor((total_kernel_area+1)//2), torch.tensor(1 / total_kernel_area), num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255, ndim=2, dim=1)

        class RoundModule(torch.nn.Module):
            def forward(self, x):
                return torch.round(x)

        round_module = RoundModule()
        output_module = torch.nn.Sequential(pool_module, mult_module, round_module, oss_module)
        output_module.scale = scale
        output_module.zero_point = zero_point
        return output_module

    @staticmethod
    def from_adaptiveavgpool2d(model, start, end):
        '''
        The below function offloads to NPU, but only works for output size->1,1
        Otherwise we will use a generic implementation, as AdaptiveAvgPooling will be used mostly only at the model end
         and is not compute intensive
        '''
        pool_module = dict(model.named_modules())[start.target]
        total_kernel_area = pool_module.output_size[0] * pool_module.output_size[1]
        if total_kernel_area != 1:
            return __class__.from_passthrough_module(model, start, end)

        scale, zero_point = __class__._get_scale_zero_point_from_previous(model, start)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
            zero_point = torch.tensor(zero_point)


        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(
            torch.tensor((total_kernel_area+1)//2), torch.tensor(1 / total_kernel_area), num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255, ndim=2, dim=1)

        class ReduceSum(torch.nn.Module):
            def forward(self, x):
                return torch.sum(x, dim=(2, 3))

        class RoundModule(torch.nn.Module):
            def forward(self, x):
                return torch.round(x)

        reducesum_module = ReduceSum()
        round_module = RoundModule()

        output_module = torch.nn.Sequential(reducesum_module, round_module, oss_module)
        output_module.scale = scale
        output_module.zero_point = zero_point
        return output_module
    # @staticmethod
    # def from_fq(model, start, end):
    #     fq_module1 = dict(model.named_modules())[start.target]
    #     fq_observed1 = fq_module1.activation_post_process
    #     scale, zero_point = fq_observed1.calculate_qparams()
    #     device = scale.device
    #     oss_offset = torch.tensor(0.0,device=device)
    #     oss_scale = torch.tensor(1.0,device=device)
    #     oss_shift = torch.tensor(1.0,device=device)
    #     oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255)
    #     return oss_module

    # @staticmethod
    # def from_fq_bn_fq(model, start, end):
    #     fq_module1 = dict(model.named_modules())[start.target]
    #     bn_module = dict(model.named_modules())[start.next.target]
    #     fq_module2 = dict(model.named_modules())[end.target]
    #
    #     zero_point_offset_for_activation = -128
    #     fq_observed1 = fq_module1.activation_post_process
    #     scale1, zero_point1 = fq_observed1.calculate_qparams()
    #
    #     bn_sigma = torch.sqrt(bn_module.running_var + bn_module.eps)
    #     bn_weight = (bn_module.weight / bn_sigma)
    #     bn_bias = bn_module.bias - (bn_module.running_mean / bn_sigma)
    #
    #     fq_observed2 = fq_module2.activation_post_process
    #     scale2, zero_point2 = fq_observed2.calculate_qparams()
    #     device = scale2.device
    #
    #     combined_weight = bn_weight / scale2
    #     oss_shift, oss_scale = compute_shift_scale(combined_weight)
    #     oss_offset = bn_bias / scale2 + zero_point2 + zero_point_offset_for_activation
    #
    #     oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127)
    #     return oss_module

    # @staticmethod
    # def from_conv_relu_fq(model, start, end):
    #     conv_module = conv_relu_module[0]
    #     weight_scale = conv_module.weight_scale
    #     weight_zero_point = conv_module.weight_zero_point
    #     device = weight_scale.device
    #     conv_int_in_float = torch.nn.Conv2d(conv_module.in_channels, conv_module.out_channels, conv_module.kernel_size)
    #     oss_module = __class__.from_fq(fq_module)
    #     relu_module = torch.nn.ReLU()
    #     return torch.nn.Sequential(conv_int_in_float, oss_module, relu_module)

    # @staticmethod
    # def from_cbn_fq(model, start, end):
    #     qconv_module = dict(model.named_modules())[start.target]
    #     fq_module = dict(model.named_modules())[end.target]
    #     fq_observed = fq_module.activation_post_process
    #     scale, zero_point = fq_observed.calculate_qparams()
    #     device = scale.device
    #     oss_offset = torch.tensor(0.0,device=device)
    #     oss_scale = torch.tensor(0.0,device=device)
    #     oss_shift = torch.tensor(0.0,device=device)
    #     conv_module = cbn_module.to_float()
    #     conv_module = torch.nn.Conv2d()
    #     return conv_module
