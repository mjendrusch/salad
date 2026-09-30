"""Efficient triangle multiplication ops."""

import jax
import jax.numpy as jnp

import haiku as hk

from salad.modules.basic import Linear

class FastTriangle(hk.Module):
    """Fast triangle multiplicative update from Wohlwend et al. 2025."""
    def __init__(self, factor=1, final_init="linear", name = None):
        super().__init__(name)
        self.factor = factor
        self.final_init = final_init

    def __call__(self, pair, mask=None):
        if mask is None:
            mask = True
        gate = Linear(pair.shape[-1] * self.factor, bias=True)(pair)
        value = Linear(pair.shape[-1] * self.factor, bias=False)(pair) * jax.nn.sigmoid(gate)
        value = jnp.where(mask[..., None], value, 0.0)
        x, y, xt, yt = jnp.moveaxis(value, -1, 0).reshape(4, -1, *value.shape[:-1])
        xt = jnp.swapaxes(xt, 1, 2)
        yt = jnp.swapaxes(yt, 1, 2)
        value = jnp.concatenate((
            jnp.einsum("...ik,...jk->...ij", x, y),
            jnp.einsum("...ik,...jk->...ij", xt, yt)), axis=0)
        value = jnp.moveaxis(value, 0, -1)
        value = hk.LayerNorm([-1], True, True)(value)
        gate = Linear(pair.shape[-1], bias=True)(value)
        value = Linear(pair.shape[-1], bias=False, initializer=self.final_init)(value) * jax.nn.sigmoid(gate)
        return value

class Triangle(hk.Module):
    """Fast triangle multiplicative update from Wohlwend et al. 2025."""
    def __init__(self, factor=1, name = None):
        super().__init__(name)
        self.factor = factor

    def __call__(self, pair, mask=None):
        if mask is None:
            mask = True
        left = (
            jax.nn.sigmoid(Linear(pair.shape[-1] * self.factor, bias=False)(pair))
            * Linear(pair.shape[-1] * self.factor, bias=False)(pair)
        )
        left = jnp.where(mask[..., None], left, 0)
        right = (
            jax.nn.sigmoid(Linear(pair.shape[-1] * self.factor, bias=False)(pair))
            * Linear(pair.shape[-1] * self.factor, bias=False)(pair)
        )
        right = jnp.where(mask[..., None], right, 0)
        gate = jax.nn.sigmoid(Linear(pair.shape[-1] * self.factor, bias=False)(pair))
        gate = jnp.where(mask[..., None], gate, 0)
        forward = gate * hk.LayerNorm([-1], True, True)(jnp.einsum("ikc,jkc->ijc", left, right))
        left = (
            jax.nn.sigmoid(Linear(pair.shape[-1] * self.factor, bias=False)(pair))
            * Linear(pair.shape[-1] * self.factor, bias=False)(pair)
        )
        left = jnp.where(mask[..., None], left, 0)
        right = (
            jax.nn.sigmoid(Linear(pair.shape[-1] * self.factor, bias=False)(pair))
            * Linear(pair.shape[-1] * self.factor, bias=False)(pair)
        )
        right = jnp.where(mask[..., None], right, 0)
        gate = jax.nn.sigmoid(Linear(pair.shape[-1] * self.factor, bias=False)(pair))
        gate = jnp.where(mask[..., None], gate, 0)
        reverse = gate * hk.LayerNorm([-1], True, True)(jnp.einsum("kic,kjc->ijc", left, right))
        return forward + reverse
