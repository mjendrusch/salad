import jax
import jax.numpy as jnp

def lennard_jones(distance, depth, radii):
    return depth * ((radii / distance) ** 12 - 2 * (radii / distance) ** 6)
