from functools import partial
from typing import Callable, Optional, List, Type

import torch
import nir
from nirtorch import extract_nir_graph

import norse.torch.module.iaf as iaf
import norse.torch.module.leaky_integrator_box as leaky_integrator_box
import norse.torch.module.lif as lif
import norse.torch.module.lif_box as lif_box

import logging

def _extract_norse_module(
    module: torch.nn.Module, dt: float = 0.001,
    post_map: Callable[[torch.nn.Module, float], nir.NIRNode] = lambda a, b : None
) -> Optional[nir.NIRNode]:
    if isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            input_shape=None,
            weight=module.weight.detach(),
            bias=module.bias.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    if isinstance(module, lif.LIFCell):
        return nir.CubaLIF(
            tau_mem=dt / module.p.tau_mem_inv.detach(),  # Invert time constant
            tau_syn=dt / module.p.tau_syn_inv.detach(),  # Invert time constant
            v_threshold=module.p.v_th.detach(),
            v_leak=module.p.v_leak.detach(),
            r=torch.ones_like(module.p.v_leak.detach()),
        )
    if isinstance(module, lif_box.LIFBoxCell):
        return nir.LIF(
            tau=dt / module.p.tau_mem_inv.detach(),  # Invert time constant
            v_threshold=module.p.v_th.detach(),
            v_leak=module.p.v_leak.detach(),
            r=torch.ones_like(module.p.v_leak.detach()),
        )
    if isinstance(module, leaky_integrator_box.LIBoxCell):
        return nir.LI(
            tau=dt / module.p.tau_mem_inv.detach(),  # Invert time constant
            v_leak=module.p.v_leak.detach(),
            r=torch.ones_like(module.p.v_leak.detach()),
        )
    if isinstance(module, iaf.IAFCell):
        return nir.IF(
            r=torch.ones_like(module.p.v_th.detach()),
            v_threshold=module.p.v_th.detach(),
        )
    if isinstance(module, torch.nn.Linear):
        if module.bias is None:  # Add zero bias if none is present
            return nir.Affine(
                module.weight.detach(), torch.zeros(*module.weight.shape[:-1])
            )
        else:
            return nir.Affine(module.weight.detach(), module.bias.detach())
    if isinstance(module, torch.nn.Flatten):
        return nir.Flatten(None, module.start_dim, module.end_dim)

    mapped_node = post_map(module, dt)
    if mapped_node == None:
        logging.warn(f"No mapping found for module of type {type(module).__name__}")
    return post_map(module, dt)


def to_nir(
    module: torch.nn.Module,
    sample_data: torch.Tensor,
    model_name: str = "norse",
    dt: float = 0.001,
    ignore_types: List[Type] = [],
    post_map: Callable[[torch.nn.Module, float], nir.NIRNode] = lambda a, b : None
) -> nir.NIRNode:
    return extract_nir_graph(
        module,
        partial(_extract_norse_module, dt=dt, post_map=post_map),
        sample_data,
        model_name=model_name,
        ignore_types=ignore_types
    )
