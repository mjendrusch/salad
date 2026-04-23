"""Diffusion models for protein architecture generation (secondary structure elements)."""

import jax
import jax.numpy as jnp
import haiku as hk

from salad.modules.basic import GatedMLP, Linear, init_linear, init_zeros, init_glorot

class ArchitectureDiffusion(hk.Module):
    def __init__(self, config, name = "architecture_diffusion"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        pass # TODO
