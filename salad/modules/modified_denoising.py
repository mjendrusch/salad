import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from salad.modules.noise_schedule_benchmark import (
    StructureDiffusionPredict, StructureDiffusionNoise)

def rot_x(angle):
    return jnp.array([
        [1.0000000,      0.0000000,      0.0000000],
        [0.0000000,  np.cos(angle), -np.sin(angle)],
        [0.0000000,  np.sin(angle),  np.cos(angle)]])
def rot_y(angle):
    return jnp.array([
        [np.cos(angle),  0.0000000, -np.sin(angle)],
        [    0.0000000,  1.0000000,      0.0000000],
        [np.sin(angle),  0.0000000,  np.cos(angle)]])
def rot_z(angle):
    return jnp.array([
        [np.cos(angle), -np.sin(angle), 0.0000000],
        [np.sin(angle),  np.cos(angle), 0.0000000],
        [0.0000000,          0.0000000, 1.0000000]])
def apply_rot(rot, x):
    return jnp.einsum("...c,cd->...d", x, rot)

def make_denoising_step(config, symmetriser):
    symmetriser = ...
    init_data = symmetriser.init_data()
    def inner(data, prev):
        denoiser = StructureDiffusionPredict(config)
        noiser = StructureDiffusionNoise(config)
        pos_noised = noiser(data["pos"])
        data["pos_noised"] = symmetriser.symmetrise_noise(pos_noised)
        result, prev = denoiser(data, prev)
        result["pos"] = symmetriser.symmetrise_pos(result["pos"])
        return result
    return init_data, jax.jit(hk.transform(inner).apply)

def centering(x):
    return x.at[:].add(-x[:, 1].mean(axis=0))
def screw_replicate(x, count=3, angle=0, radius=15.0, translation=0.0):
    rotmat = jnp.array([
        [np.cos(angle),  0.0000000, -np.sin(angle)],
        [    0.0000000,  1.0000000,      0.0000000],
        [np.sin(angle),  0.0000000,  np.cos(angle)]])
    rotmat_x = [jnp.eye(3), rotmat]
    for idx in range(count - 2):
        rotmat_x.append(jnp.einsum("ac,cb->ab", rotmat_x[-1], rotmat))
    size = x.shape[0] // count
    idx = count // 2
    first = jnp.einsum(
        "...c,cd->...d",
        centering(x[idx * size:(idx + 1) * size]),
        rotmat_x[idx].T)
    first = first.at[:].add(jnp.array([radius, 0.0, 0.0]))
    replicates = []
    for idx in range(count):
        replicates.append(
            jnp.einsum("...c,cd->...d", first, rotmat_x[idx])
          + idx * jnp.array([0.0, translation, 0.0]))
    result = jnp.concatenate(replicates, axis=0)
    return result - result[:, 1].mean(axis=0)
