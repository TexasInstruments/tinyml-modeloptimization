import torch
from torch.fx import GraphModule, Node, symbolic_trace
from typing import Dict, List

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
    if not isinstance(offset, torch.Tensor):
        offset = torch.tensor(offset)
        weight = torch.tensor(weight)
    weight_abs = weight.abs()
    weight_sign = weight.sign()
    scale_max = (2**num_bits_scale)-1
    power_of_2 = torch.floor(torch.log2(
        scale_max/weight_abs))*torch.tensor([1.0])
    shift = power_of_2.clamp(min=0, max=((2**num_bits_shift)-1))
    scale = weight_abs * torch.pow(torch.tensor([2.0]), shift)

    mask = torch.isnan(scale)
    scale[mask] = 0
    shift[mask] = 1

    if torch.sum(scale > scale_max) != 0:
        raise RuntimeError(
            f"Error in Quant convert:compute_offset_scale_shift. Output multiplication could not be converted. \n"
            f"Invalid in output multiplication value: {weight.cpu().detach().numpy()} \n"
            f"Make sure that the model is trained properly with good hyper parameters. "
            f"(try adjusting: training epochs, learning rate, QAT after float training etc): \n"
        )

    scale = weight_sign*torch.round(scale)
    shift_mult = torch.pow(torch.tensor([2.0]), -shift)

    if print_mse:
        weight_hat = scale * torch.pow(torch.tensor([2.0]), -shift)
        mse = torch.mean((weight-weight_hat)**2)
        print(mse)

    # add round offset to the offset. since the offset is before the scale, divide it by scale before adding
    shift_round_offset = torch.pow(torch.tensor([2.0]), (shift-1)) / scale
    offset = torch.round(offset + shift_round_offset)
    return offset, scale, shift_mult


def _get_parent_name(target: str):
    '''Gets the name of the parent module and attribute name of the module from the target of the module'''
    *parent, name = target.rsplit('.', 1)
    return (parent[0] if parent else ''), name


def get_scale_from_parent(main_modules: Dict, node: Node) -> float:
    if node.op != 'root':
        if hasattr(main_modules[node.target], 'scale'):
            return main_modules[node.target].scale
        else:
            return get_scale_from_parent(main_modules, node._prev)
    else:
        return 1


def get_scale(main_modules: Dict, start: Node) -> float:
    new_scale = 1
    if hasattr(start, 'target') and hasattr(main_modules[start.target], 'scale'):
        new_scale = main_modules[start.target].scale
    else:
        new_scale = get_scale_from_parent(main_modules, start)
    return new_scale


def find_hanging_nodes(main_module: GraphModule) -> List[Node]:
    count = []
    for node in main_module.graph.nodes:
        if (node.op not in ('output', 'placeholder') and len(node.users) == 0):
            count.append(node)
    return count


def remove_hanging_nodes(main_module: GraphModule) -> None:

    while True:
        hanging_nodes = find_hanging_nodes(main_module)
        if len(hanging_nodes) == 0:
            break
        for node in hanging_nodes:
            main_module.graph.erase_node(node)

    main_module.graph.lint()
    main_module.recompile()

    return


def remove_intermediate_call_modules(main_module: GraphModule, new_node: Node, start: Node, end: Node) -> None:
    main_modules = dict(main_module.named_modules())
    ptr = start
    while ptr != end:
        if ptr.op == 'call_module':
            parent_name, name = _get_parent_name(ptr.target)
            parent_module = main_modules[parent_name]
            parent_module.__delattr__(name)

        temp = ptr.next
        ptr.replace_all_uses_with(new_node)
        main_module.graph.erase_node(ptr)
        ptr = temp

    if ptr.op == 'call_module':
        parent_name, name = _get_parent_name(end.target)
        parent_module = main_modules[parent_name]
        parent_module.__delattr__(name)

    ptr.replace_all_uses_with(new_node)
    main_module.graph.erase_node(end)

    return


def replace_call_function_or_method(main_module: GraphModule, start: torch.Node, end: torch.Node, replace_module: torch.nn.Module, module_no: int=0) -> None:

    if start == end:
        traced_replacement = symbolic_trace(replace_module)
        replacement_nodes = [node for node in traced_replacement.graph.nodes if node.op not in ['placeholder', 'output']]

        if len(replacement_nodes) == 1:

            # call_function or call_method operation
            replacement_operation = replacement_nodes[0].op
            # function call or method name
            function_or_method = replacement_nodes[0].target
            # Replacing in main_module graph specifying insert point after start within this scope
            with main_module.graph.inserting_after(start):
                # Insert a new node (replacement_node) using 'call_method' or 'call_function'
                new_node = getattr(main_module.graph, replacement_operation)(function_or_method, start.args, start.kwargs)
                # Replaces nodes that used the value of 'start' to now use that value new_node
                start.replace_all_uses_with(new_node)
            # Remove the unused 'start' node from graph as 'new_node' has replaced it
            main_module.graph.erase_node(start)
            main_module.recompile()

            return

    # Get the name of replaced module
    new_node_name = 'replaced_' + str(replace_module.__class__.__name__) + '_' + str(module_no)
    # Add the child module in main_module
    main_module.add_module(new_node_name, replace_module)

    # Inserting in main_module graph specifying insert point before start within this scope
    with main_module.graph.inserting_before(start):
        # Collect all the args which aren't attribute and needs to passed to call module
        args = []
        for arg in start.args:
            if type(arg) == Node and arg.op != "get_attr":
                args.append(arg)

        new_node = main_module.graph.call_module(new_node_name, tuple(args), {})
        # Remove all the intermediate call module nodes
        remove_intermediate_call_modules(main_module, new_node, start, end)

    return


def replace_call_module(main_module: GraphModule, start: Node, end: Node, replace_module: torch.nn.Module, module_no: int=0) -> None:

    main_modules = dict(main_module.named_modules())

    # Get the parent module name and attribute name
    parent_name, attr_name = _get_parent_name(start.target)
    parent_module = main_modules[parent_name]
    # Set the attribute of parent module with the replacement module
    parent_module.__setattr__(attr_name, replace_module)
    # Get the scale of the parent module
    # new_scale = get_scale(main_modules, start)
    # Store the scale of the parent module
    # scales_of_nodes[start.name] = new_scale

    # If there are more nodes between start and end, remove them all
    if start != end:
        # Initialize pointers for iteration
        new_node = start
        # Remove all the intermediate call module nodes
        remove_intermediate_call_modules(main_module, new_node, start.next, end)
    main_module.graph.lint()
    main_module.recompile()
    remove_hanging_nodes(main_module)
    return

class ReduceSum(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=(2, 3))
            
class RoundModule(torch.nn.Module):
    def forward(self, x):
        return torch.round(x)
            
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
            self.register_buffer('offset', offset.reshape(1, -1, 1, 1))
            self.register_buffer('mult', mult.reshape(1, -1, 1, 1))
            self.register_buffer('shift_mult', shift_mult.reshape(1, -1, 1, 1))
        elif ndim == 2 and dim == 1:
            self.register_buffer('offset', offset.reshape(1, -1))
            self.register_buffer('mult', mult.reshape(1, -1))
            self.register_buffer('shift_mult', shift_mult.reshape(1, -1))
        elif ndim == 1:
            self.register_buffer('offset', offset.reshape(-1))
            self.register_buffer('mult', mult.reshape(-1))
            self.register_buffer('shift_mult', shift_mult.reshape(-1))
        else:
            raise RuntimeError('Invalid dimensions')
        #

    def extra_repr(self):
        return f'offset={self.offset}, mult={self.mult}, shift={self.shift_mult}, quant_min={self.quant_min}, quant_max={self.quant_max}'

    def forward(self, x):
        y = (x + self.offset) * self.mult
        y = y * self.shift_mult
        if self.use_floor:
            # the floor operation mimics the actual shift and bit select in hardware
            y = torch.floor(y).clamp(min=self.quant_min, max=self.quant_max)
        else:
            y = torch.round(y).clamp(min=self.quant_min, max=self.quant_max)
        return y

class TINPUQuantizedReplacement:
    @staticmethod
    def from_child_module(model, start, end, module_no=0):
        '''
        copy quant scale, zero_point from child module
        '''
        named_modules = dict(model.named_modules())
        module = named_modules[start.target]
        replace_module = module
        named_children = list(module.named_children())
        if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
            replace_module = module
        elif len(named_children) > 0:
            last_child_name, last_child = named_children[-1]
            if hasattr(last_child, 'scale') and hasattr(last_child, 'zero_point'):
                module.scale = last_child.scale
                module.zero_point = last_child.zero_point
                replace_module = module

        replace_call_module(model, start, end, replace_module, module_no)

        return None

    @staticmethod
    def _get_scale_zero_point_attrs_from_model(model, node):
        # Get scale, zero point of node from model.node_scale_0, model.node_zero_point_0
        scale = zero_point = None
        node_name_base = node.name.replace('.', '_')
        # Attribute names of scale and zero point of node
        scale_attr_name, zero_point_attr_name = node_name_base + '_scale_0', node_name_base + '_zero_point_0'
        if hasattr(model, scale_attr_name) and hasattr(model, zero_point_attr_name):
            scale, zero_point = getattr(model, scale_attr_name), getattr(model, zero_point_attr_name)
        return scale, zero_point

    @staticmethod
    def _get_scale_zero_point_from_previous(model: GraphModule, node: Node):
        
        scale = zero_point = None
        named_modules = dict(model.named_modules())
        prev_node = node.prev

        while prev_node:
            if prev_node.target in named_modules:
                previous_module = named_modules[prev_node.target]
                #Return the scale and zero_point of previous module
                if hasattr(previous_module, 'scale') and hasattr(previous_module, 'zero_point'):
                    scale, zero_point = previous_module.scale, previous_module.zero_point
                return scale, zero_point
            else:
                # Return the scale and zero_point of previous node from model
                scale, zero_point = __class__._get_scale_zero_point_attrs_from_model(model, prev_node)
                if scale is not None and zero_point is not None:
                    return scale, zero_point
            prev_node = prev_node.prev
        
        return scale, zero_point

    @staticmethod
    def from_q(model, start, end, module_no=0):
        offset_scale_shift = [torch.tensor([0]), torch.tensor([1]), torch.tensor([1])]
        oss_module = TINPUOffsetScaleShift(*offset_scale_shift, -128, 127, ndim=4, dim=1)
        oss_module.scale = 1.0
        oss_module.zero_point = 0.0
        # Replace quantization method with OSS Layer
        replace_call_function_or_method(model, start, end, oss_module, module_no)
        return None

    @staticmethod
    def from_q_id(model, start, end, module_no=0):
        # Quantization Node
        q_node = start
        scale = getattr(model, q_node.args[1].target)
        zero_point = getattr(model, q_node.args[2].target)
        # OSS Module
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(zero_point*0.0, 1/scale, num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=4, dim=1)
        oss_module.scale = scale
        oss_module.zero_point = zero_point
        # Replace quantize function with OSS Module
        replace_call_function_or_method(model, start, end, oss_module, module_no)
        return None

    @staticmethod
    def from_q_qbn(model, start, end, module_no=0):
        # Quantized Batch Normalization Module
        qbn_module = dict(model.named_modules())[end.target]
        bn_sigma = torch.sqrt(qbn_module.running_var + qbn_module.eps)

        scale2 = qbn_module.scale
        zero_point2 = qbn_module.zero_point

        oss_offset = (- qbn_module.running_mean + qbn_module.bias*bn_sigma)
        # first get the effective weight due to batchnorm
        combined_weight = (qbn_module.weight / bn_sigma)
        # then modify the weight by output scale so that the output is converted to output scale
        combined_weight = combined_weight / scale2
        # OSS Module
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(oss_offset, combined_weight, num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127, ndim=4, dim=1)
        oss_module.scale = scale2
        oss_module.zero_point = zero_point2
        # Remove the scale, zero_point, quantize method and bn layer with OSS Module
        replace_call_function_or_method(model, start, end, oss_module, module_no)
        return None

    @staticmethod
    def from_module_with_dq(model, start, end, module_no=0):
        module = dict(model.named_modules())[start.target]
        scale, zero_point = __class__._get_scale_zero_point_from_previous(model, start.next)
        mult_module = MultiplyModule(scale)
        mult_module.scale = 1.0
        mult_module.zero_point = 0.0
        seq_module = torch.nn.Sequential(module, mult_module)
        seq_module.scale = mult_module.scale
        seq_module.zero_point = mult_module.zero_point
        return seq_module

    @staticmethod
    def from_module_with_q(model, start, end, module_no=0):
        module = dict(model.named_modules())[start.target]
        oss_module = __class__.from_q(model, start.next, end, module_no)
        seq_module = torch.nn.Sequential(module, oss_module)
        seq_module.scale = oss_module.scale
        seq_module.zero_point = oss_module.zero_point
        return seq_module

    @staticmethod
    def from_qconv_relu(model, start, end, module_no=0, with_relu=True):
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
        
        seq_module.scale = qconvrelu_module.scale
        seq_module.zero_point = qconvrelu_module.zero_point

        replace_call_module(model, start, end, seq_module, module_no)

        return None

    @staticmethod
    def from_qlinear(model, start, end, module_no=0, with_relu=False):
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
        replace_call_module(model, start, end, seq_module, module_no)
        return None

    @staticmethod
    def from_qlinear_relu(model, start, end, module_no=0, with_relu=True):
        return __class__.from_qlinear(model, start, end, module_no, with_relu=with_relu)

    @staticmethod
    def from_passthrough_module(model, start, end, module_no=0):
        named_modules = dict(model.named_modules())
        passthrough_module = named_modules[start.target]
        replace_module = passthrough_module
        scale, zero_point = __class__._get_scale_zero_point_attrs_from_model(model, start)
        if hasattr(passthrough_module, 'scale') and hasattr(passthrough_module, 'zero_point'):
            replace_module = passthrough_module
        elif scale is not None and zero_point is not None:
            passthrough_module.scale = scale
            passthrough_module.zero_point = zero_point
            replace_module = passthrough_module
        elif start.prev.target in named_modules:
            prev_module = named_modules[start.prev.target]
            passthrough_module.scale = prev_module.scale
            passthrough_module.zero_point = prev_module.zero_point
            replace_module = passthrough_module
        elif scale is not None and zero_point is not None:
            scale, zero_point = __class__._get_scale_zero_point_attrs_from_model(model, start.prev)
            passthrough_module.scale = scale
            passthrough_module.zero_point = zero_point
            replace_module = passthrough_module

        replace_call_module(model, start, end, replace_module, module_no)
        return None
    
    @staticmethod
    def from_flatten(model, start, end, module_no=0):
        named_modules = dict(model.named_modules())
        replace_module = torch.nn.Flatten(*start.args[1:])
        if start.target in named_modules:
            replace_module = named_modules[start.target]
        quantization_node = start.next.next.next
        scale = getattr(model, quantization_node.args[1].target)
        zero_point = getattr(model, quantization_node.args[2].target)
        replace_module.scale = scale
        replace_module.zero_point = zero_point

        replace_call_function_or_method(model, start, quantization_node, replace_module, module_no)
        return None

    @staticmethod
    def from_dq(model, start, end, module_no=0):
        # if start.next.target != 'output':
        #     return __class__.from_dq_with_dq(model, start, end, module_no)
        id_module = torch.nn.Identity()
        id_module.scale = 1.0
        id_module.zero_point = 0.0
        replace_call_function_or_method(model, start, end, id_module, module_no)
        return None

    @staticmethod
    def from_dq_with_dq(model, start, end, module_no=0):
        scale, zero_point = __class__._get_scale_zero_point_from_previous(model, start)
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(zero_point*0.0, scale, num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255, ndim=4, dim=1)
        oss_module.scale = scale
        oss_module.zero_point = zero_point
        replace_call_function_or_method(model, start, end, oss_module, module_no)
        return None

    @staticmethod
    def from_avgpool2d(model, start, end, module_no=0):
        # AvgPool2D module
        pool_module = dict(model.named_modules())[start.target]
        # OSS Module
        scale, zero_point = __class__._get_scale_zero_point_from_previous(model, start)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
            zero_point = torch.tensor(zero_point)
        # Calculate the total_kernel_area from pool module kernel size
        total_kernel_area = pool_module.kernel_size[0] * pool_module.kernel_size[1]
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(torch.tensor((total_kernel_area+1)//2), torch.tensor(1 / total_kernel_area), num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255, ndim=2, dim=1)
        # Multiply Module
        mult_module = MultiplyModule(total_kernel_area)
        # Round Module
        round_module = RoundModule()
        # Sequential Module comprising of AvgPool2D, Multiply, Round, OSS
        output_module = torch.nn.Sequential(pool_module, mult_module, round_module, oss_module)
        output_module.scale = scale
        output_module.zero_point = zero_point
        # Replace AvgPool2D with Sequential Module
        replace_call_module(model, start, end, output_module, module_no)
        
        return None
    
    @staticmethod
    def from_adaptiveavgpool2d(model, start, end, module_no=0):
        '''
        The below function offloads to NPU, but only works for output size->1,1
        Otherwise we will use a generic implementation, as AdaptiveAvgPooling will be used mostly only at the model end
        and is not compute intensive
        '''
        # AdaptiveAvgPool2D Module
        pool_module = dict(model.named_modules())[start.target]
        # OSS Module
        scale, zero_point = __class__._get_scale_zero_point_from_previous(model, start)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
            zero_point = torch.tensor(zero_point)
        # Calculate the total_kernel_area from pool module output size
        total_kernel_area = pool_module.output_size[0] * pool_module.output_size[1]
        if total_kernel_area != 1:
            #  If output size isn't (1, 1), we will use the generic implementation
            return __class__.from_passthrough_module(model, start, end, module_no)
        oss_offset, oss_scale, oss_shift = compute_offset_scale_shift(torch.tensor((total_kernel_area+1)//2), torch.tensor(1 / total_kernel_area), num_bits_scale=8)
        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255, ndim=2, dim=1)
        # Round Module
        round_module = RoundModule()
        # ReduceSum Module
        reduce_sum_module = ReduceSum()
        # Sequential Module comprising of Reduce, Round, OSS
        output_module = torch.nn.Sequential(reduce_sum_module, round_module, oss_module)
        output_module.scale = scale
        output_module.zero_point = zero_point
        # Replace AdaptiveAvgPool2
        replace_call_module(model, start, end, output_module, module_no)
        return None
    