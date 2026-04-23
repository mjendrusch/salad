import jax
import jax.numpy as jnp

from salad.aflib.model.geometry import Vec3Array, Rot3Array

def random_rotation(key):
    key_x, key_y = jax.random.split(key, 2)
    cx = Vec3Array.from_array(jax.random.normal(key_x, (3,)))
    cy = Vec3Array.from_array(jax.random.normal(key_y, (3,)))
    rot = Rot3Array.from_two_vectors(cx, cy).to_array()
    return rot

def augment_rotation(key, pos):
    rot = random_rotation(key)
    if isinstance(pos, list):
        pos = [jnp.einsum("cd,...c->...d", rot, i) for i in pos]
    else:
        pos = jnp.einsum("cd,...c->...d", rot, pos)
    return pos

def augment_index(key, index, size=1_000):
    offset = jax.random.randint(key, (), minval=-size, maxval=size)
    return index + offset

def augment_category(key, index, num_classes=1_000):
    categories = jax.random.permutation(key, num_classes)
    return categories[index]
