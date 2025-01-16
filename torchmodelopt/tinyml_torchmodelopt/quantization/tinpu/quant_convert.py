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


from typing import Union, Dict, Any
from inspect import isfunction, ismethod
import torch

import warnings
import torch
import types
from inspect import getmodule, isfunction
from torch import nn, fx, Tensor
from torch.fx import symbolic_trace, GraphModule, Node
from typing import Dict, Any, Union, List
import operator
import copy
from copy import deepcopy
from types import FunctionType

# from . import custom_modules

# from .custom_symbolic_trace import custom_symbolic_trace
custom_symbolic_trace = symbolic_trace
'''
this module's function are implemented to changes nodes only.
no change is made on the incoming arguments and keyword arguments.
to do so self-made surgery functions are required.
for example, custom_surgery_functions module can be checked
'''


# Wrapper module for modules in nn package
class InstaModule(nn.Module):
    def __init__(self,preDefinedLayer:nn.Module) -> None:
        super().__init__()
        self.model=preDefinedLayer

    def forward(self,x):
        return self.model(x)


def _get_parent_name(target:str):
    '''gets the name of the parent module and attribute name of the module from the target of the module'''
    *parent, name = target.rsplit('.', 1)
    return ( parent[0] if parent else ''), name


def remove_identiy(model:nn.Module, verbose_mode=False, **kwargs):
    # removed due to RuntimeError
    # RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.
    # model=deepcopy(model)

    traced_model=custom_symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    modules= dict(traced_model.named_modules())
    n=0
    nodes=[]
    for node in traced_model.graph.nodes:
        if (node.op == 'call_module') and isinstance(modules[node.target],nn.Identity):
                nodes.append(node)
    for node in nodes:
        try:
            node.replace_all_uses_with(node.args[0])
            copy_found=False
            for node_1 in nodes:
                if node!=node_1 and node.target==node_1.target:
                    copy_found=True
                    break
            if not copy_found:
                parent_name,name=_get_parent_name(node.target)
                modules[parent_name].__delattr__(name)
                modules.pop(node.target)
            traced_model.graph.erase_node(node)
            n+=1
        except Exception as e:
            if verbose_mode:
                print(n,e)
    traced_model.graph.lint()
    traced_model.recompile()
    if verbose_mode:
        print('Identity removed',n)
    return traced_model


def _get_parent_name(target: str):
    '''gets the name of the parent module and attribute name of the module from the target of the module'''
    *parent, name = target.rsplit('.', 1)
    return (parent[0] if parent else ''), name


def replace_module_nodes(model, pattern, replacement, copy_args=[], verbose_mode: bool = False):
    '''replaces a  modules of pattern type to replacement module in the module structure'''
    modules = dict(model.named_modules())
    replace_obj = replacement() if type(replacement) == type else replacement
    if type(pattern) != type:
        pattern_type = type(pattern)
    else:
        pattern_type = pattern

    n = 0
    for key_name, module in modules.items():
        if isinstance(module, pattern_type):
            n += 1
            parent_name, name = _get_parent_name(key_name)
            replace_obj = copy.deepcopy(replace_obj)
            for copy_arg in copy_args:
                if hasattr(module, copy_arg):
                    setattr(replace_obj, copy_arg, getattr(module, copy_arg))
            # modules[key_name] = replace_obj
            modules[parent_name].__setattr__(name, replace_obj)
    if verbose_mode:
        print(pattern_type.__name__, n)


def replace_function_nodes(model, pattern_function, replacement, verbose_mode=False, **kwargs):
    '''replaces a call function node to node with replacement function '''
    traced_model = custom_symbolic_trace(copy.deepcopy(model))
    no_of_module = 0
    n = 0
    if isfunction(replacement) or type(replacement).__name__ in ('builtin_function_or_method', 'function'):
        for node in traced_model.graph.nodes:
            if node.target == pattern_function:
                kwargs = node.kwargs if kwargs is None else kwargs
                with traced_model.graph.inserting_before(node):
                    new_node = traced_model.graph.call_function(replacement, node.args, kwargs)
                    node.replace_all_uses_with(new_node)
                    traced_model.graph.erase_node(node)
                    n += 1
    else:
        for node in traced_model.graph.nodes:
            if node.target == pattern_function:
                kwargs = node.kwargs if kwargs is None else kwargs
                with traced_model.graph.inserting_before(node):
                    replace_obj = replacement(**kwargs) if type(replacement) == type else replacement
                    if not isinstance(replace_obj, nn.Module):
                        return traced_model
                    new_node_name = type(replace_obj).__name__ + str(no_of_module)
                    n += 1
                    no_of_module += 1
                    traced_model.add_submodule(new_node_name, copy.deepcopy(replace_obj))
                    args = []
                    for arg in node.args:
                        if type(arg) == Node:
                            if arg.op != "get_attr":
                                args.append(arg)
                    new_node = traced_model.graph.call_module(new_node_name, tuple(args), {})
                    node.replace_all_uses_with(new_node)
                    # Remove the old node from the graph
                    traced_model.graph.erase_node(node)
    traced_model.graph.lint()
    traced_model.recompile()
    _remove_hanging_nodes(main_module=traced_model)
    if verbose_mode:
        print(pattern_function, str(n + no_of_module))
    return traced_model


# checks whether two nodes are  equal or not
def _are_both_node_equal(first_node: Node, second_node: Node, first_graph_module: Union[GraphModule, None] = None,
                         second_graph_module: Union[GraphModule, None] = None):
    '''
    checks whether two nodes are equal or not.
    till now for two node to be same they must have same operation
    '''

    # for operator and torch function counter-parts
    operationDict = {torch.add: operator.add, torch.sub: operator.sub, torch.mul: operator.mul, operator.add: torch.add,
                     operator.sub: torch.sub, operator.mul: torch.mul}

    if first_node.op == second_node.op:
        if first_node.op == 'placeholder':
            # for placeholder both should have same number of users
            return len(first_node.users) == len(second_node.users)

        if first_node.op == 'output':
            # for output both should have same number of argument i.e. number of outputs are same
            return (len(first_node.args) + len(first_node.kwargs)) == (len(second_node.args) + len(second_node.kwargs))

        if first_node.op == 'call_function':
            if first_node.target == second_node.target:
                # if both refer to same function
                return True

            elif first_node.target in operationDict.keys():
                # if it is one  of add, sub, mul from either of operator module or torch module it should be the counter part
                return second_node.target == operationDict[first_node.target]

            else:
                return False

        if first_node.op == 'call_method':
            # both should refer to same method
            return first_node.target == second_node.target
        if first_node.op == 'get_attr':
            _, target1 = _get_parent_name(first_node.target)
            _, target2 = _get_parent_name(second_node.target)
            return target1 == target2

        if (first_graph_module == None) or (second_graph_module == None):
            raise RuntimeError(
                "GraphModules are required for both nodes\nas at least one of them is 'call_module' node.")

        # for call_module node
        modules_in_1st_graph = dict(first_graph_module.named_modules())
        modules_in_2nd_graph = dict(second_graph_module.named_modules())
        module_1 = modules_in_1st_graph[first_node.target]
        module_2 = modules_in_2nd_graph[second_node.target]
        # TODO change equality module fpr hyperparameters
        # both should be instances of same class
        return str(type(module_1)) == str(type(module_2))

    # TODO add for comparing two nodes if they refer to a pair functional and module node

    return False


# searches pattern with on single input and one output other wise a single node with out checking kwargs
def straight_chain_searcher(main_module: GraphModule, pattern_module: GraphModule):
    '''
    searches for straight pattern matches in node list of the graph

    it only allows:
        i)  if pattern has one input and one output
        ii) if pattern has only one node other than placeholders or output

    '''

    main_module_nodes: list[fx.Node] = list(main_module.graph.nodes)
    pattern_module_nodes = []
    number_of_input = 0
    number_of_output = 0

    # removes placeholders or outputs
    for node in pattern_module.graph.nodes:
        if node.op == 'placeholder':
            number_of_input += 1
        elif node.op == 'output':
            number_of_output = len(node.args)
        else:
            pattern_module_nodes.append(node)
    main_module_node_num = len(main_module_nodes)
    pattern_module_node_num = len(pattern_module_nodes)

    assert (number_of_input == 1 and number_of_output == 1) or pattern_module_node_num == 1, \
        'This function is not for multi-input or multi-output'

    # similar approach to searching pattern in an list
    main_index = 0
    patt_index = 0
    matched = list()
    second_start_index = -1
    inp, out = -1, -1
    while (main_index < main_module_node_num):
        main_node = main_module_nodes[main_index]
        patn_node = pattern_module_nodes[patt_index]
        cond = _are_both_node_equal(main_node, patn_node, main_module, pattern_module)
        if cond:
            if main_node == pattern_module_nodes[0] and second_start_index == -1 and patt_index != 0:
                second_start_index = main_index
            if patt_index == 0:
                inp = main_node
            if patt_index == (pattern_module_node_num - 1):
                out = main_node
                matched.append((inp, out))
                second_start_index = -1
            patt_index += 1
            patt_index = patt_index % pattern_module_node_num
            main_index += 1
        else:
            inp, out = -1, -1
            patt_index = 0
            if second_start_index == -1:
                main_index += 1
            else:
                main_index = second_start_index
                second_start_index = -1

    return matched


def straight_type_chain_searcher(main_module: GraphModule, type_pattern: List):
    '''
    searches for straight pattern of type of module matches in node list of the graph

    '''
    # it only allows:
    #     i)  if pattern has one input and one output
    #     ii) if pattern has only one node other than placeholders or output

    main_module_nodes: list[fx.Node] = list(main_module.graph.nodes)
    modules_in_main_graph = dict(main_module.named_modules())
    main_module_node_num = len(main_module_nodes)
    pattern_type_num = len(type_pattern)

    assert isinstance(type_pattern, list) and all(
        isinstance(typ, (type, str,)) or isinstance(typ, (types.FunctionType, types.BuiltinFunctionType)) for typ in
        type_pattern), \
        'This function only supports searching for a straight sequence of types of module!'

    # similar approach to searching pattern in an list
    main_index = 0
    patt_index = 0
    matched = list()
    second_start_index = -1
    inp, out = -1, -1

    operationDict = {torch.add: operator.add, torch.sub: operator.sub, torch.mul: operator.mul, operator.add: torch.add,
                     operator.sub: torch.sub, operator.mul: torch.mul}

    def are_both_function_equal(first_function, second_function):
        if first_function == second_function:
            # if both refer to same function
            return True
        elif hasattr(first_function, 'target') and first_function.target in operationDict.keys():
            # if it is one  of add, sub, mul from either of operator module or torch module it should be the counter part
            return second_function == operationDict[first_function]
        elif first_function in operationDict.keys():
            # if it is one  of add, sub, mul from either of operator module or torch module it should be the counter part
            return second_function == operationDict[first_function]

        else:
            return False

    while (main_index < main_module_node_num):
        main_node = main_module_nodes[main_index]
        patn_type = type_pattern[patt_index]
        # cond = _are_both_node_equal(main_node, patn_type, main_module, pattern_module)
        cond = (main_node.op == 'call_module' and isinstance(patn_type, type) and isinstance(
            modules_in_main_graph[main_node.target], patn_type)) \
               or (main_node.op == 'call_method' and isinstance(patn_type, str) and main_node.target == patn_type) \
               or (main_node.op == 'call_function' and isinstance(patn_type, (
        types.FunctionType, types.BuiltinFunctionType)) and are_both_function_equal(main_node.target, patn_type))
        if cond:
            if main_node == type_pattern[0] and second_start_index == -1 and patt_index != 0:
                second_start_index = main_index
            if patt_index == 0:
                inp = main_node
            if patt_index == (pattern_type_num - 1):
                out = main_node
                matched.append((inp, out))
                second_start_index = -1
            patt_index += 1
            patt_index = patt_index % pattern_type_num
            main_index += 1
        else:
            inp, out = -1, -1
            patt_index = 0
            if second_start_index == -1:
                main_index += 1
            else:
                main_index = second_start_index
                second_start_index = -1

    return matched


# replaces a pattern fom start to end in graph module to a call_module node
def _replace_pattern(main_module: GraphModule, start: Node, end: Node, replace_module: nn.Module,
                     no_of_module_replaced: int = 0):
    '''
    replaces nodes from start node to end node with the replacement module

    if pattern has only a single operational node start and end wull be same
    then, it will check for replacement if it also has a single node.
    if it has only a single operational node, it will check if it is call function or call method node.
    if it is so, it will add respective node not a call module node
    '''

    main_modules = dict(main_module.named_modules())

    # if pattern has a single operational node only
    if start == end:
        # if start is a call function or call method node
        if start.op in ['call_function', 'call_method']:
            traced_replacement = custom_symbolic_trace(replace_module)
            replacement_nodes = []

            # removes all placeholders and output node
            for node in traced_replacement.graph.nodes:
                if node.op not in ['placeholder', 'output']:
                    replacement_nodes.append(node)

            # if replacement has single operational node
            if len(replacement_nodes) == 1:
                # if operation of replacement is call function
                if replacement_nodes[0].op == 'call_function':
                    with main_module.graph.inserting_after(start):
                        new_node = main_module.graph.call_function(replacement_nodes[0].target, start.args,
                                                                   start.kwargs)
                        start.replace_all_uses_with(new_node)
                    main_module.graph.erase_node(start)
                    main_module.recompile()
                    return
                elif replacement_nodes[0].op == 'call_method':
                    # if operation of replacement is call method
                    with main_module.graph.inserting_after(start):
                        new_node = main_module.graph.call_method(replacement_nodes[0].target, start.args, start.kwargs)
                        start.replace_all_uses_with(new_node)
                    main_module.graph.erase_node(start)
                    main_module.recompile()
                    return

    # for call module start node, even pattern and replacement have a single operational node each
    if start.op == 'call_module':
        parent_name, name = _get_parent_name(start.target)
        parent_module = main_modules[parent_name]
        parent_module.__setattr__(name, replace_module)
        # main_modules[start.target] = replace_module
        new_node = start
        ptr = start.next

        # if pattern has more than one operational nodes, removes all extra nodes
        if start != end:
            while ptr != end:
                if ptr.op == 'call_module':
                    parent_name, name = _get_parent_name(ptr.target)
                    parent_module = main_modules[parent_name]
                    parent_module.__delattr__(name)

                ptr.replace_all_uses_with(new_node)
                temp = ptr.next
                main_module.graph.erase_node(ptr)
                ptr = temp

            if ptr.op == 'call_module':
                parent_name, name = _get_parent_name(end.target)
                parent_module = main_modules[parent_name]
                parent_module.__delattr__(name)

            ptr.replace_all_uses_with(new_node)
            main_module.graph.erase_node(end)

    # if start of the pattern is call function or call method node,
    # even if replacement has single call_module node,
    # but pattern must have more than one operational nodes
    else:
        # creates a module for replacement and adds it to main model
        new_node_name = 'replaced_' + str(replace_module.__class__.__name__) + '_' + str(no_of_module_replaced)
        main_module.add_module(new_node_name, replace_module)

        # creates new node for replacement and deletes all nodes from start to end
        with main_module.graph.inserting_before(start):
            args = []
            for arg in start.args:
                if type(arg) == Node:
                    if arg.op != "get_attr":
                        args.append(arg)

            new_node = main_module.graph.call_module(new_node_name, tuple(args), {})
            ptr = start
            while ptr != end:
                if ptr.op == 'call_module':
                    parent_name, name = _get_parent_name(ptr.target)
                    parent_module = main_modules[parent_name]
                    parent_module.__delattr__(name)

                ptr.replace_all_uses_with(new_node)
                temp = ptr.next
                main_module.graph.erase_node(ptr)
                ptr = temp

            if ptr.op == 'call_module':
                parent_name, name = _get_parent_name(end.target)
                parent_module = main_modules[parent_name]
                parent_module.__delattr__(name)

            ptr.replace_all_uses_with(new_node)
            main_module.graph.erase_node(end)
        # main_modules.update({new_node_name:replace_module})
    main_module.graph.lint()
    main_module.recompile()
    _remove_hanging_nodes(main_module)


def _remove_hanging_nodes(main_module: GraphModule):
    def find_hanging_nodes(main_module: GraphModule):
        count = []
        for node in main_module.graph.nodes:
            if (node.op not in ('output', 'placeholder') and len(node.users) == 0):
                count.append(node)
        return count

    h_nodes = find_hanging_nodes(main_module)
    while len(h_nodes) > 0:
        for node in h_nodes:
            main_module.graph.erase_node(node)
        h_nodes = find_hanging_nodes(main_module)
    main_module.graph.lint()
    main_module.recompile()


# replaces all matches with call_module node
def _replace_all_matches(main_module: GraphModule, matches, replace_module: nn.Module):
    '''
    replace all pattern from matches with a copy of replacement module
    '''

    no_of_module_replaced = 0
    for (start, end) in matches:
        replace_module_copy = copy.deepcopy(replace_module)
        _replace_pattern(main_module, start, end, replace_module_copy, no_of_module_replaced)
        no_of_module_replaced += 1


# replace nodes if they don't need any change with their keyword arguments and arguements
def graph_pattern_replacer(main_module: Union[GraphModule, nn.Module, callable],
                           pattern_module: Union[GraphModule, nn.Module, callable],
                           replace_module: Union[GraphModule, nn.Module, callable], verbose_mode=False):
    '''
    searches for all matches in the graph and replaces all of them with replacement module
    '''
    replace_module = replace_module() if type(replace_module) == type else replace_module

    if not isinstance(main_module, GraphModule):
        main_module = custom_symbolic_trace(main_module)
    if not isinstance(pattern_module, GraphModule):
        pattern_module = custom_symbolic_trace(pattern_module)

    pattern_nodes = []
    number_of_input = 0
    number_of_output = 0
    for node in pattern_module.graph.nodes:
        if node.op == 'placeholder':
            number_of_input += 1
        elif node.op == 'output':
            # TODO: should it be += ?
            number_of_output = len(node.args)
        else:
            pattern_nodes.append(node)

    # pattern should have a single node or (single input and single output)
    if (number_of_input == 1 and number_of_output == 1) or (len(pattern_nodes) == 1):
        matches = straight_chain_searcher(main_module, pattern_module)
        _replace_all_matches(main_module, matches, replace_module)
        if verbose_mode:
            print(type(pattern_module).__name__, len(matches))
    else:
        warnings.warn(''' unable to change model as pattern does n't satisfy for the criteria of pattern searcher''')

    return main_module


def _is_replacable(pattern: Union[GraphModule, nn.Module, callable]):
    try:
        if not isinstance(pattern, GraphModule):
            pattern = custom_symbolic_trace(pattern)
    except:
        return False
    return True


def _replace_unsupported_layers(model: nn.Module, example_inputs: list = None, example_kwargs: dict = None,
                                replacement_dict: Dict[Any, Union[nn.Module, callable]] = None,
                                copy_args: list = [], verbose_mode: bool = False, **kwargs) -> GraphModule | nn.Module:
    '''
    main function that does the surgery

    it does default surgery if no replacement dictionry is given
    replacement dictionary may contain
    keys                value
    callable        ->  callable            : any call function to call_function if they take same argument partial agument may -                                         be used
    callable        ->  nn.Module           : any call function to call_function if they take same argument partial agument may -                                         be used
    Any             ->  Callable            : any self-made surgery function
    nn.Module       ->  nn.Module           : any nn.Module pattern to replace with another nn.Module
    type            ->  type/nn.Module      : replaces sub-module of same type as patttern using traditional python approach
    '''

    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
    for pattern, replacement in replacement_dict.items():
        if pattern is None:
            continue

        if isfunction(pattern) or type(pattern).__name__ in ('builtin_function_or_method', 'function'):
            # replacement must be partially defined function or work with same args and kwargs
            if isinstance(replacement, (list, tuple)):
                kwargs = replacement[1] if len(replacement) > 1 else None
                replacement = replacement[0]
            else:
                kwargs = dict()
            model = replace_function_nodes(model, pattern, replacement, verbose_mode=verbose_mode, **kwargs)
        elif isfunction(replacement) or ismethod(replacement):
            # for self-made surgery function
            model = replacement(model, pattern=pattern, example_inputs=example_inputs, verbose_mode=verbose_mode)
        else:
            # class of MOdule of
            if isinstance(pattern, type):
                replace_module_nodes(model, pattern, replacement, copy_args=copy_args, verbose_mode=verbose_mode)
            else:
                # for nn.Module
                if pattern.__class__.__name__ in dir(torch.nn):
                    # if the pattern is present in nn directory,
                    # a wrapper module is required, for successful
                    # surgery on that module
                    model = graph_pattern_replacer(model, pattern, replacement, verbose_mode=verbose_mode)
                    pattern = InstaModule(pattern)

                # calls the main surgery function
                model = graph_pattern_replacer(model, pattern, replacement, verbose_mode=verbose_mode)
    model = remove_identiy(model)
    model.delete_all_unused_submodules()

    return model


# Default Flags for replacement dict
# for custom replacement add a custom flag name (any string not in default flag) as key and map it to a dict containing pattern and replacement
# note if same key is used for two pattern the last replacement will be performed
default_replacement_flag_dict = {
}

# Default Flags for replacement dict with no training required as subset of default flags
# for custom replacement add a custom flag name (any string not in default flag) as key and map it to a dict containing pattern and replacement
# note if same key is used for two pattern the last replacement will be performed
default_replacement_flag_dict_no_training = {
}

# Mapping between the flags and the actual replacements corresponding to them
# This dictionary is used whenever a flag is enabled to fetch the corresponding replacement entries
flag_to_dict_entries = {
}


# returns default dictionary for replacement
def get_replacement_flag_dict_default(return_flags=True, can_retrain=True):
    '''
    returns the default flag dictionary.
    to see the dict print 'default_replacement_flag_dict' from the file this function is in
    '''
    flag_dict = default_replacement_flag_dict if can_retrain else default_replacement_flag_dict_no_training
    if return_flags:
        return flag_dict
    replacement_entries_dict = {}
    for k, v in flag_dict.items():
        if k in flag_to_dict_entries and v in (True, False):
            if v:
                v = flag_to_dict_entries[k]
            else:
                continue
        else:
            continue
        replacement_entries_dict.update({k, v})
    return replacement_entries_dict


def get_replacement_dict(
        replacement_flag_dict: dict[
            str | nn.Module | FunctionType | type, bool | nn.Module | FunctionType | type | tuple[
                FunctionType, FunctionType]] = None,
        can_retrain: bool = True,
):
    '''
    this function actually converts the flags mapped to True to their corresponding replacements
    if no replacement_flag_dict is given it uses default flag dictionary based on can_retrain
    if the flags is not registered in 'flag_to_dict_entries', its value should be a dict of replacement and that will be updated in the dictionary
    '''

    if can_retrain:
        replacement_flag_dict = replacement_flag_dict or default_replacement_flag_dict
    else:
        replacement_flag_dict = replacement_flag_dict or default_replacement_flag_dict_no_training

    replacement_dict: dict[Any, list[tuple]] = {}

    replacement_dict = {}
    for k, v in replacement_flag_dict.items():
        if k in flag_to_dict_entries and v in (True, False):
            if v:
                v = flag_to_dict_entries[k]
            else:
                continue
        else:
            if not isinstance(v, dict):
                warnings.warn(
                    f'if {k} is not a default flag or its value is not a boolean, the value must be a dict. So, this entry will be discarded!')
                continue
        replacement_dict.update(v)
    return replacement_dict


def replace_unsupported_layers(model: nn.Module, example_inputs: list = None, example_kwargs: dict = None,
                               replacement_dict: Dict[Any, Union[nn.Module, callable]] = None,
                               copy_args: list = [], can_retrain=True, verbose_mode: bool = False, **kwargs):
    # TODO write appropiate documentation for this function

    '''
    wrapper to the function that does the surgery

    it does default surgery if no replacement dictionry is given
    replacement dictionry must contains flag name as keys and True/False or a replacement dictionary corresponding to flag as value

    behavior for each value:
    value               behavior
    True            ->  convert according to mapped replacement dict
    False           ->  discard it
    dict            ->  update the main replacement dictionary with the entries of it

    values for replacement dict
    keys                value
    callable        ->  callable            : any call function to call_function if they take same argument partial agument may be used
    callable        ->  nn.Module           : any call function to call_function if they take same argument partial agument may be used
    Any             ->  Callable            : any self-made surgery function
    nn.Module       ->  nn.Module/type      : any nn.Module pattern to replace with another nn.Module
    type            ->  type/nn.Module      : replaces sub-module of same type as patttern using traditional python approach
    '''
    example_inputs = example_inputs if example_inputs is not None else []
    example_kwargs = example_kwargs or {}
    # TODO Check for functions
    if model.training:
        RuntimeWarning("The model is in train mode, converting to eval mode. This might change the network behaviour.")
        model.eval()
        is_train_mode = True
    else:
        is_train_mode = False

    replacement_dict = get_replacement_dict(replacement_dict, can_retrain=can_retrain)

    model = deepcopy(model)

    final_model = _replace_unsupported_layers(model, example_inputs, example_kwargs, replacement_dict, copy_args,
                                              verbose_mode, **kwargs)

    if is_train_mode:
        final_model.train()

    return final_model

