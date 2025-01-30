from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk

from salad.modules.basic import MLP, init_relu

def resi_dual(local, incremental, output):
    local = hk.LayerNorm([-1], True, True)(local + output)
    incremental = incremental + output
    return local, incremental

def resi_dual_input(features):
    return features[0]

def prenorm_skip(local, output):
    return local + output

def postnorm_skip(local, output):
    return hk.LayerNorm([-1], True, True)(local + output)

def prenorm_input(local):
    return hk.LayerNorm([-1], True, True)(local)

def drop(data, p=0.1, is_training=True):
    # FIXME: this should 100% be "if is_training"!
    if is_training:
        mask = jax.random.bernoulli(
            hk.next_rng_key(), p, data.shape)
        data = jnp.where(mask, 0, data) / (1 - p)
    return data

class Transition(hk.Module):
    def __init__(self, factor=4, depth=2,
                 activation=jax.nn.relu,
                 final_init="zeros",
                 name: Optional[str] = "transition"):
        super().__init__(name)
        self.activation = activation
        self.final_init = final_init
        self.factor = factor
        self.depth = depth

    def __call__(self, inputs):
        normalized = hk.LayerNorm(axis=[-1], create_scale=True, create_offset=True)(inputs)
        update = MLP(
            inputs.shape[-1] * self.factor,
            out_size=inputs.shape[-1],
            activation=self.activation,
            depth=self.depth, init=init_relu(),
            final_init=self.final_init)(normalized)
        return update

