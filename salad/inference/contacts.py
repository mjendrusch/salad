import jax
import jax.numpy as jnp

from salad.modules.utils.geometry import index_mean

def compact_step(positions, index, lr=1e-3):
    ca = positions[:, 1]
    update = index_mean(ca, index) - ca
    return positions + lr * update[:, None, :]

def clash_loss(positions):
    return ... # TODO

def clash_step(positions, lr=1e-2):
    return ... # TODO
