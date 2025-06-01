"""Efficient triangle multiplication ops."""

import jax
import jax.numpy as jnp

import haiku as hk

from salad.modules.basic import Linear

class FastTriangle(hk.Module):
    """Fast triangle multiplicative update from Wohlwend et al. 2025."""
    def __init__(self, factor=1, name = None):
        super().__init__(name)
        self.factor = factor

    def __call__(self, pair, mask=None):
        if mask is None:
            mask = True
        gate = Linear(pair.shape[-1], bias=False, initializer="relu")(pair)
        value = Linear(pair.shape[-1], bias=False)(pair) * jax.nn.gelu(gate)
        value = jnp.where(mask, value, 0.0)
        x, y, xt, yt = jnp.moveaxis(value, -1, 0).reshape(4, -1, value.shape[:-1])
        xt = jnp.swapaxis(xt, 1, 2)
        yt = jnp.swapaxis(yt, 1, 2)
        value = jnp.concatenate((
            jnp.einsum("...ik,...jk->...ij", x, y),
            jnp.einsum("...ik,...jk->...ij", xt, yt)), axis=0)
        value = jnp.moveaxis(value, 0, -1)
        value = hk.LayerNorm([-1], True, True)(value)
        gate = Linear(pair.shape[-1], bias=False, initializer="relu")(value)
        value = Linear(pair.shape[-1], bias=False)(value) * gate
        return value
