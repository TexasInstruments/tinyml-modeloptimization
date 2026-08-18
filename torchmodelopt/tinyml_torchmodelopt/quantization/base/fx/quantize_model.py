#################################################################################
# Copyright (c) 2018-2025, Texas Instruments Incorporated - http://www.ti.com
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
from torch.ao.quantization import quantize_fx, QConfig


def apply_quantization_to_supported_layers(qconfig_mapping, model):
    """Remove quantization from layers that are not Conv, BatchNorm, Linear, or Pooling.

    This function keeps the global qconfig but disables it for unsupported layer types,
    restricting quantization to commonly quantizable layers. Only leaf (non-container)
    modules are checked - container modules are allowed to propagate quantization to
    their children.

    For ConvTranspose modules, applies ch_axis=1 since their weight layout differs from
    standard Conv modules (output channels are in dimension 1 instead of 0).

    Args:
        qconfig_mapping: QConfigMapping instance to be configured
        model: Model to iterate over

    Returns:
        Modified QConfigMapping with unsupported layers set to None
    """
    supported_types = (
        torch.nn.Identity, torch.nn.Dropout,
        torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
        torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.Linear,
        torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d,
        torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d,
        torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool3d,
        torch.nn.AdaptiveMaxPool1d, torch.nn.AdaptiveMaxPool2d, torch.nn.AdaptiveMaxPool3d,
    )

    convtranspose_types = (
        torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
    )

    # Get the global qconfig
    global_qconfig = qconfig_mapping.global_qconfig

    # Recursively check modules and set unsupported leaf modules to None
    for name, module in model.named_modules():
        if name == '':
            continue

        # Only check leaf modules (modules with no children)
        if list(module.children()):
            continue

        # Set qconfig to None for unsupported leaf modules
        if not isinstance(module, supported_types):
            qconfig_mapping.set_module_name(name, None)
        # For ConvTranspose modules, apply qconfig with ch_axis=1
        elif isinstance(module, convtranspose_types) and global_qconfig is not None:
            qconfig_with_ch_axis_1 = _get_qconfig_with_ch_axis(global_qconfig, ch_axis=1)
            qconfig_mapping.set_module_name(name, qconfig_with_ch_axis_1)

    return qconfig_mapping


def _get_qconfig_with_ch_axis(qconfig, ch_axis):
    """Create a new QConfig with modified ch_axis for weight observer.

    Args:
        qconfig: Original QConfig
        ch_axis: Channel axis value to set

    Returns:
        New QConfig with weight observer modified to use the specified ch_axis
    """
    if qconfig is None:
        return None

    # Extract weight and activation fake quantize objects
    weight_fake_quant = qconfig.weight
    activation_fake_quant = qconfig.activation

    # Create new weight fake quantize with ch_axis parameter
    if weight_fake_quant is not None:
        weight_fake_quant = weight_fake_quant.with_args(ch_axis=ch_axis)

    # Create new QConfig with modified weight fake quantize
    return QConfig(weight=weight_fake_quant, activation=activation_fake_quant)


def _has_batch_norm_after_observer(model, observer_node):
    """Check if batch norm exists in the data flow after observer node.

    Traverses the graph from observer node to find batch norm layers,
    accounting for QuantizeLinear/DequantizeLinear operations.

    Args:
        model: The model (GraphModule)
        observer_node: The observer node to check from

    Returns:
        bool: True if batch norm is found in the data flow
    """
    visited = set()
    to_visit = list(observer_node.users.keys())
    named_modules = dict(model.named_modules())

    while to_visit:
        node = to_visit.pop(0)
        if node in visited:
            continue
        visited.add(node)

        if node.op == 'call_module':
            module = named_modules.get(node.target)
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Identity)):
                return True
            # Continue traversing through non-BN modules
            to_visit.extend(node.users.keys())
        elif node.op == 'call_function':
            # Skip through function calls (e.g., quantize/dequantize operations)
            to_visit.extend(node.users.keys())

    return False


def remove_input_observer_before_bn(model):
    """Remove observer attached to input placeholder node.

    After FX model preparation, an observer is attached to the input placeholder
    (activation_post_process_0). This removes that observer and rewires the graph
    to pass input directly to the first layer, avoiding QuantizeLinear/DequantizeLinear
    on raw inputs.

    Only removes the observer if there is a batch normalization after it.
    The graph is recompiled to maintain consistency after rewiring.

    Args:
        model: The prepared GraphModule
    """
    if not hasattr(model, 'graph'):
        return

    # Find the first placeholder node (model input)
    placeholder_node = None
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            placeholder_node = node
            break

    if placeholder_node is None:
        return

    # Find observer call immediately after placeholder
    observer_node = None
    for user in placeholder_node.users:
        if user.op == 'call_module' and 'activation_post_process' in user.target:
            observer_node = user
            break

    if observer_node is None:
        return

    # Check if observer is followed by batch normalization (possibly through QDQ operations)
    has_bn_after = _has_batch_norm_after_observer(model, observer_node)

    # Only remove observer if batch norm follows it
    if not has_bn_after:
        return

    # Collect users first to avoid modification during iteration
    observer_users = list(observer_node.users.keys())

    # Rewire users of observer to use placeholder directly
    for observer_user in observer_users:
        observer_user.replace_input_with(observer_node, placeholder_node)

    # Remove the observer node from graph
    model.graph.erase_node(observer_node)

    # Remove observer module from model
    if hasattr(model, observer_node.target):
        delattr(model, observer_node.target)

    # Recompile to ensure consistency
    model.graph.lint()
    model.recompile()


def prepare_quantized_model(model, qconfig_mapping, example_inputs, is_qat):
    """Prepare model for quantization with FX graph mode.

    Applies quantization configuration to supported layers, prepares the model
    for either QAT or PTQ, and removes input observers.

    Args:
        model: PyTorch model to quantize
        qconfig_mapping: QConfigMapping with quantization configuration
        example_inputs: Example input tensor for model tracing
        is_qat: If True, use prepare_qat_fx; if False, use prepare_fx

    Returns:
        Prepared GraphModule ready for quantization
    """
    # Apply quantization only to supported layers
    qconfig_mapping = apply_quantization_to_supported_layers(qconfig_mapping, model)

    # Prepare model for quantization
    if is_qat:
        prepared_model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
    else:
        prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)

    # Remove input observer to avoid quantization on raw inputs
    remove_input_observer_before_bn(prepared_model)

    return prepared_model
