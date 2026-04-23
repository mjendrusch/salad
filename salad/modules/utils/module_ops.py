"""This module provides utilities for working with Haiku modules."""

from typing import Dict, Any

import jax
import haiku as hk

def stop_parameter_gradient(model: hk.Module):
    """Stop gradients to the parameters of an input `model`.
    
    Example:
    .. code-block:: python
        @hk.transform
        def run():
            x = jnp.ones((10, 16))
            model = hk.Linear(32)
            return stop_parameter_gradient(model)(x).mean()
        params = run.init(key)
        gradient_norm = (jax.grad(run.apply, argnums=(0,))(params)[0] ** 2).sum()
        print("should be zero", gradient_norm)
    """
    def inner(*inputs, **kwargs):
        if hk.running_init():
            return model(*inputs, **kwargs)
        else:
            model_fn = hk.transform(model)
            params_dict = model.params_dict()
            params = _unflatten_params(params_dict)
            return model_fn.apply(
                jax.lax.stop_gradient(params),
                hk.next_rng_key(), *inputs, **kwargs)
    return inner

def _unflatten_params_obsolete(params: Dict[str, jax.Array], sep="/") -> Dict[str, Any]:
    key_order = dict()
    for name in params:
        key_view = key_order
        hierarchy = name.split(sep)
        for idx, subname in enumerate(hierarchy):
            if subname not in key_view:
                key_view[subname] = dict()
            if idx == len(hierarchy) - 1:
                key_view[subname] = params[name]
                key_view = key_order
            else:
                key_view = key_view[subname]
    return key_order

def _unflatten_params(params, sep="/"):
    key_order = dict()
    for name in params:
        *base, param_name = name.split(sep)
        base = "/".join(base)
        if base not in key_order:
            key_order[base] = dict()
        key_order[base][param_name] = params[name]
    return key_order

