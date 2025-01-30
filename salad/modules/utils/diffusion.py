from typing import Optional, Union
import jax
import jax.numpy as jnp
from alphafold.model.geometry import Vec3Array, Rigid3Array, Rot3Array

from salad.modules.utils.geometry import index_mean, index_count

Float = Union[float, jnp.ndarray]

def diffuse_sequence(key, x: jnp.ndarray, t: Float, mask_index=20):
    corrupt_aa = jax.random.bernoulli(key, p=t, shape=x.shape)
    out = jnp.where(corrupt_aa, mask_index, x)
    return out, corrupt_aa

def diffuse_coordinates(key, x: Vec3Array, t: float, scale=1.0) -> Vec3Array:
    target = Vec3Array.from_array(jax.random.normal(key, list(x.shape) + [3]))
    target = target * scale
    return jnp.sqrt(t) * target + jnp.sqrt(1 - t) * x

def diffuse_coordinates_vp(key, x: Vec3Array, mask, batch_index, t: float, scale=10.0,
                           min_sigma=0.0, return_eps=False) -> Vec3Array:
    target = scale * jax.random.normal(key, [x.shape[0], x.shape[1], 3])
    center = index_mean(target[:, 1], batch_index, mask)
    target = target - center[:, None, :]
    target = Vec3Array.from_array(target)
    target = target
    s = 0.01
    alpha_bar = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar = jnp.clip(alpha_bar, 0, 1)
    alpha = jnp.sqrt(alpha_bar)
    sigma = jnp.sqrt(1 - alpha_bar)
    sigma = jnp.clip(sigma, min_sigma, jnp.inf)
    target = sigma * target
    result = x * alpha + target
    if return_eps:
        eps = target
        return result, eps, alpha
    return result

def diffuse_coordinates_blend(key, x: Vec3Array, mask, batch_index, t: float, scale=10.0):
    target = scale * jax.random.normal(key, [x.shape[0], x.shape[1], 3])
    center = index_mean(target[:, 1], batch_index, mask[:, None])
    target = target - center[:, None, :]
    target = Vec3Array.from_array(target)
    return t * target + (1 - t) * x

def diffuse_coordinates_vp_scaled(key, x: Vec3Array, mask, batch_index, t: float, scale=None,
                                  min_sigma=0.0, rescale=1.0, return_eps=False,
                                  truncate=1.0) -> Vec3Array:
    x = x.to_array()
    x_sigma = jnp.sqrt(
        index_mean(x[:, 1] ** 2, batch_index, mask[..., None]).mean(axis=1)
      - index_mean(x[:, 1], batch_index, mask[..., None]).mean(axis=1) ** 2)
    x = Vec3Array.from_array(x)
    if scale is None:
        scale = x_sigma[:, None, None]
    target = rescale * scale * jax.random.normal(key, [x.shape[0], x.shape[1], 3])
    center = jnp.zeros_like(target)[:, 1].at[batch_index].add(target[:, 1])
    center /= jnp.maximum(jnp.zeros_like(target)[:, 1].at[batch_index].add(1.0), 1e-6)
    center = center[batch_index]
    target = target - center[:, None, :]
    target = Vec3Array.from_array(target)
    target = target
    s = 0.01
    alpha_bar = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar = jnp.clip(alpha_bar, 0, 1)
    alpha = jnp.sqrt(alpha_bar)
    sigma = jnp.sqrt(1 - alpha_bar)
    sigma = jnp.clip(sigma, min_sigma, jnp.inf)
    target = sigma * target
    target = target * truncate
    result = x * alpha + target
    if return_eps:
        eps = target
        return result, eps, alpha
    return result

def diffuse_coordinates_chroma(key, x: Vec3Array, mask: jnp.ndarray,
                               batch_index: jnp.ndarray, t: float,
                               return_eps=False) -> Vec3Array:
    x = x.to_array()
    Rg2 = ((x[:, 1] - index_mean(x[:, 1], batch_index, mask[..., None])) ** 2).sum(axis=-1)
    Rg = jnp.sqrt(jnp.maximum(index_mean(Rg2, batch_index, mask), 1e-6))
    b = ... # TODO
    x = Vec3Array.from_array(x)
    gaussian_chain_std = ... # TODO
    gaussian_chain_decay = ... # TODO
    z = jax.random.normal(key, (x.shape[0], x.shape[1], 3))
    def correlate_body(carry, z):
        next = carry.mean(axis=-2, keepdims=True) * gaussian_chain_decay + z * gaussian_chain_std
        return next, next
    # TODO

def diffuse_coordinates_edm(key, x: Vec3Array, batch_index, t: jnp.ndarray, symm=False) -> Vec3Array:
    target = jax.random.normal(key, [x.shape[0], x.shape[1], 3])
    center = jnp.zeros_like(target)[:, 1].at[batch_index].add(target[:, 1])
    center /= jnp.maximum(jnp.zeros_like(target)[:, 1].at[batch_index].add(1.0), 1e-6)
    center = center[batch_index]
    target = target - center[:, None, :]
    if symm:
        rotmat = jnp.array([[-0.5000000,  0.0000000,  0.8660254],
                  [0.0000000,  1.0000000,  0.0000000],
                  [-0.8660254,  0.0000000, -0.5000000]])
        replicate = target[:100]
        # target = jnp.concatenate((replicate - 0.1, replicate, replicate + 0.1), axis=0)
        # replicate -= index_mean(replicate[:, 1], batch_index[:100], mask[:100, None])[:, None]
        plus_1 = jnp.einsum("...c,cd->...d", replicate, rotmat)
        plus_2 = jnp.einsum("...c,cd->...d", plus_1, rotmat)
        target = jnp.concatenate((replicate, plus_1, plus_2), axis=0)
        # target = target.at[100:200].set(target[:100] + 2)
        # target = target.at[200:].set(target[:100] - 2)
    target = Vec3Array.from_array(target)
    target = t * target
    result = x + target
    return result

def position_std(x, index, mask):
    ca = x[:, 1]
    center = index_mean(ca, index, mask[..., None])
    ca -= center
    return jnp.sqrt(
        index_mean(ca ** 2, index, mask[..., None]).mean(axis=1)
      - index_mean(ca, index, mask[..., None]).mean(axis=1) ** 2)

# def diffuse_atom_cloud(key, x: Vec3Array, mask: jnp.ndarray, batch_index: jnp.ndarray,
#                        t: jnp.ndarray, cloud_std=None,
#                        residue_std=4.0, correlated_residues=False,
#                        symm=None) -> Vec3Array:
#     if cloud_std is None:
#         # compute standard deviation of atom positions in the structure
#         x_sigma = position_std(x.to_array(), batch_index, mask)
#         cloud_std = x_sigma[:, None, None]
#     # if we want correlated residues
#     if correlated_residues:
#         cloud_key, residue_key = jax.random.split(key, num = 2)
#         target = cloud_std * jax.random.normal(cloud_key, [x.shape[0], 1, 3])
#         target += residue_std * jax.random.normal(residue_key, [x.shape[0], x.shape[1], 3])
#     # otherwise
#     else:
#         target = cloud_std * jax.random.normal(key, [x.shape[0], x.shape[1], 3])
#     center = jnp.zeros_like(target)[:, 1].at[batch_index].add(target[:, 1])
#     center /= jnp.maximum(jnp.zeros_like(target)[:, 1].at[batch_index].add(1.0), 1e-6)
#     center = center[batch_index]
#     target = target - center[:, None, :]
#     if symm is not None:
#         target = symm(target)
#     target = Vec3Array.from_array(target)
#     target = target
#     result = x * jnp.sqrt(1 - t ** 2) + target * t
#     return result

def diffuse_atom_cloud(key, x: Vec3Array, mask: jnp.ndarray, batch_index: jnp.ndarray,
                       t: jnp.ndarray, cloud_std=None, flow=False,
                       residue_std=4.0, correlated_residues=False,
                       symm=None) -> Vec3Array:
    if cloud_std is None:
        # compute standard deviation of atom positions in the structure
        x_sigma = position_std(x.to_array(), batch_index, mask)
        cloud_std = x_sigma[:, None, None]
    # if we want correlated residues
    if correlated_residues:
        cloud_key, residue_key = jax.random.split(key, num = 2)
        target = cloud_std * jax.random.normal(cloud_key, [x.shape[0], 1, 3])
        target += residue_std * jax.random.normal(residue_key, [x.shape[0], x.shape[1], 3])
    # otherwise
    else:
        target = cloud_std * jax.random.normal(key, [x.shape[0], x.shape[1], 3])
    # FIXME: noise is already centered. What does post-centering even do?
    center = jnp.zeros_like(target)[:, 1].at[batch_index].add(target[:, 1])
    center /= jnp.maximum(jnp.zeros_like(target)[:, 1].at[batch_index].add(1.0), 1e-6)
    center = center[batch_index]
    target = target - center[:, None, :]
    if symm is not None:
        target = symm(target)
    target = Vec3Array.from_array(target)
    target = target
    if flow:
        result = x * (1 - t) + target * t
    else:
        result = x * jnp.sqrt(1 - t ** 2) + target * t
    return result

def diffuse_target_centered(key, x: Vec3Array, is_target, hotspots,
                            mask: jnp.ndarray, batch_index: jnp.ndarray,
                            t: jnp.ndarray, cloud_std=None) -> Vec3Array:
    noise = cloud_std * jax.random.normal(key, [x.shape[0], x.shape[1], 3])
    # center noise on hotspot
    # remove center of the noise cloud
    center = jnp.zeros_like(noise)[:, 1].at[batch_index].add(noise[:, 1])
    center /= jnp.maximum(jnp.zeros_like(noise)[:, 1].at[batch_index].add(1.0), 1e-6)
    center = center[batch_index]
    noise = noise - center[:, None, :]
    # and recenter it on a hotspot residue
    ca = x[:, 1].to_array()
    noise_center = index_mean(
        ca, batch_index, (mask * hotspots)[:, None], apply_mask=False)
    noise = Vec3Array.from_array(noise)
    noise_center = Vec3Array.from_array(noise_center)
    # interpolate (VP)
    # de-yeeted version
    result = (x - noise_center[:, None]) * jnp.sqrt(1 - t ** 2) + noise * t + noise_center[:, None]
    # and fix the target
    result = Vec3Array.from_array(jnp.where(
        is_target[:, None, None], x.to_array(), result.to_array()))
    return result

def diffuse_binder_centered(key, x: Vec3Array, is_target, hotspots,
                            mask: jnp.ndarray, batch_index: jnp.ndarray,
                            t: jnp.ndarray, binder_center=None, cloud_std=None) -> Vec3Array:
    key, subkey = jax.random.split(key, 2)
    noise = cloud_std * jax.random.normal(key, [x.shape[0], x.shape[1], 3])
    # center noise on hotspot
    # remove center of the noise cloud
    center = jnp.zeros_like(noise)[:, 1].at[batch_index].add(noise[:, 1])
    center /= jnp.maximum(jnp.zeros_like(noise)[:, 1].at[batch_index].add(1.0), 1e-6)
    center = center[batch_index]
    noise = noise - center[:, None, :]
    # and recenter it on a hotspot residue
    ca = x[:, 1].to_array()
    if binder_center is not None:
        noise_center = binder_center[None]
    else:
        noise_center = index_mean(
            ca, batch_index, (~is_target)[:, None], apply_mask=False)
        # jitter the center by a couple angstroms
        # during training for the model to learn to move
        # things around
        noise_center += jax.random.normal(subkey, (3,))
    noise_center = Vec3Array.from_array(noise_center)
    noise = Vec3Array.from_array(noise)
    # interpolate (VP)
    result = (x - noise_center[:, None]) * jnp.sqrt(1 - t ** 2) + noise * t + noise_center[:, None]
    # and fix the target
    result = Vec3Array.from_array(jnp.where(
        is_target[:, None, None], x.to_array(), result.to_array()))
    return result

def diffuse_atom_chain_cloud(key, x: Vec3Array, mask: jnp.ndarray,
                             chain_index: jnp.ndarray,
                             batch_index: jnp.ndarray,
                             t: jnp.ndarray, cloud_std=None, complex_std=None,
                             std_scale_min=0.9, std_scale_max=1.1, flow=False,
                             residue_std=4.0, correlated_residues=False,
                             symm=False):
    if cloud_std is None and complex_std is None:
        cloud_std = position_std(x.to_array(), chain_index, mask)
        raw_complex_std = position_std(x.to_array(), batch_index, mask)
        key, subkey = jax.random.split(key, num = 2)
        factor = jax.random.uniform(subkey, cloud_std.shape,
                                    minval=std_scale_min, maxval=std_scale_max)[batch_index]
        raw_std = jnp.sqrt(index_mean(cloud_std ** 2, batch_index, mask))
        complex_std = jnp.sqrt(jnp.maximum(raw_complex_std ** 2 - raw_std ** 2, 1e-6))
        cloud_std = (factor * cloud_std)[:, None, None]
        complex_std = (factor * complex_std)[:, None, None]
    chain_key, complex_key = jax.random.split(key)
    target = cloud_std * jax.random.normal(chain_key, [x.shape[0], x.shape[1], 3])
    complex_noise = jax.random.normal(complex_key, [x.shape[0], x.shape[1], 3])
    target += (complex_noise * jnp.zeros_like(complex_std).at[chain_index].max(complex_std))[chain_index]
    target = Vec3Array.from_array(target)
    result = x * jnp.sqrt(1 - t ** 2) + target * t
    return result

def diffuse_chain_cloud(key, x: Vec3Array, mask: jnp.ndarray, chain_index: jnp.ndarray,
                        batch_index: jnp.ndarray, t: jnp.ndarray,
                        center_std=None, chain_std=None, residue_std=4.0,
                        correlated_residues=False) -> Vec3Array:
    if center_std is None:
        x_tmp = x.to_array()
        batch_std = position_std(x_tmp[:, 1], batch_index, mask[..., None])
        chain_std = position_std(x_tmp[:, 1], chain_index, mask[..., None])
        chain_std = chain_std[:, None, None]
        center_std = jnp.sqrt(batch_std ** 2 - chain_std ** 2)
    chain_key, center_key, residue_key, move_key = jax.random.split(key, num = 4)
    
    # randomly select whether to move chains or not
    chain_center = index_mean(x[:, 1].to_array(), batch_index, mask[..., None])
    move_chains = jax.random.bernoulli(move_key, 0.5)[batch_index]
    interpolate_center = jnp.where(
        move_chains[..., None, None],
        jax.random.normal(center_key, [x.shape[0], 1, 3])[chain_index],
        chain_center
    )

    # center the target at the selected chain center positions
    target = interpolate_center
    if correlated_residues:
        # add noise with the desired chain standard deviation
        target += chain_std * jax.random.normal(
            chain_key, [x.shape[0], 1, 3])
        # add noise at each residue position
        target += residue_std * jax.random.normal(
            residue_key, [x.shape[0], x.shape[1], 3])
    else:
        # add noise with the desired chain standard deviation
        target += chain_std * jax.random.normal(
            chain_key, [x.shape[0], x.shape[1], 3])
    target = Vec3Array.from_array(target)
    # interpolate between noise-free structure and noised structure
    result = x * jnp.sqrt(1 - t ** 2) + target * t
    return result

def diffuse_tl_R_body(carry, data):
    z, noise, new_noise, count, chain_index, batch_index = data
    s0 = 3.8 * 0.5
    sigma = 0.395 * count ** (3/5) + 7.257
    combined_sigma = jnp.sqrt(1 / (1 / (sigma ** 2) + 1 / (s0 ** 2)))
    previous, prev_chain, prev_batch = carry
    new_chain = (prev_chain != chain_index) + (prev_batch != batch_index)
    new_chain = new_chain > 0
    mean = (previous * sigma ** 2) / (s0 ** 2 + sigma ** 2)
    z = combined_sigma * z + mean - previous
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    z = z * 3.8
    new = jnp.where(new_chain, new_noise, previous + z)
    out = new[None, :] + noise
    return (new, chain_index, batch_index), out

def diffuse_tl_R(key, x, chain_index, batch_index, t: float, position_scale=10.0, return_eps=False):
    k0, k1, k2 = jax.random.split(key, 3)
    z = jax.random.normal(k0, (x.shape[0], 3,))
    noise = jax.random.normal(k1, (x.shape[0], x.shape[-1], 3))
    new_noise = jax.random.normal(k2, (x.shape[0], 3,))
    count = jnp.zeros(x.shape[0]).at[chain_index].add(1)[chain_index]
    _, target = jax.lax.scan(diffuse_tl_R_body, (jnp.zeros((3,)), -1, -1),
                         (z, noise, new_noise, count, chain_index, batch_index))
    center = jnp.zeros_like(target)[:, 1].at[batch_index].add(target[:, 1])
    center /= jnp.maximum(jnp.zeros_like(target)[:, 1].at[batch_index].add(1.0), 1e-6)
    center = center[batch_index]
    target = target - center[:, None, :]
    target = Vec3Array.from_array(target.astype(jnp.float32))
    s = 0.01
    alpha_bar = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
    alpha = jnp.sqrt(alpha_bar)
    sigma = jnp.sqrt(1 - alpha_bar)
    result = x * alpha + sigma * target
    result = result / position_scale
    if return_eps:
        eps = target
        return result, eps, alpha
    return result

def diffuse_features(key, x: jnp.ndarray, t: float, scale=4.0) -> jnp.ndarray:
    target = jax.random.normal(key, x.shape) * scale
    return jnp.sqrt(t) * target + jnp.sqrt(1 - t) * x

def diffuse_features_edm(key, x: jnp.ndarray, t: float) -> jnp.ndarray:
    target = jax.random.normal(key, x.shape)
    result = x + t * target
    return result

def diffuse_features_vp(key, x: jnp.ndarray, t: float, scale=1.0) -> jnp.ndarray:
    target = jax.random.normal(key, x.shape) * scale
    s = 0.01
    alpha_bar = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
    alpha = jnp.sqrt(alpha_bar)
    sigma = jnp.sqrt(1 - alpha_bar)
    result = x * alpha + sigma * target
    return result

def diffuse_features_permute(key, x: jnp.ndarray, t: float, mask: jnp.ndarray, scale=1.0) -> jnp.ndarray:
    noise_key, permute_key, choose_permuted_key = jax.random.split(key, num = 3)
    target = jax.random.normal(noise_key, x.shape) * scale
    x_permute = jax.lax.stop_gradient(x)
    x_permute = jnp.where(mask[..., None], x_permute, target)
    permuted_x = jax.random.permutation(permute_key, x_permute, axis=0, independent=False)
    choose_permuted = jax.random.bernoulli(choose_permuted_key, 0.15, (x_permute.shape[0],))
    target = jnp.where(choose_permuted[..., None], permuted_x, target)
    # s = 0.01
    # alpha_bar = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
    # alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
    # alpha = jnp.sqrt(alpha_bar)
    # sigma = jnp.sqrt(1 - alpha_bar)
    # FIXME:
    sigma = t
    alpha = jnp.sqrt(1 - t ** 2)
    result = x * alpha + sigma * target
    return result

def time_embedding(t, size=128, start=0, stop=1) -> jnp.ndarray:
    bins = jnp.arange(size) / size + 1 / (2 * size)
    bins = bins * (stop - start) + start
    return jnp.exp(-size * (t[..., None] - bins) ** 2)

def fourier_time_embedding(t, size=128) -> jnp.ndarray:
    exponent = 2 * jnp.arange(0, size // 2, dtype=t.dtype) / size#1_000
    denominator = 10_000 ** exponent
    result = (t[..., None]) / denominator
    return jnp.concatenate((jnp.sin(result), jnp.cos(result)), axis=-1)

def log_sigma_embedding(log_sigma, size=128, min=-4, max=0) -> jnp.ndarray:
    effective_size = size - 1
    step = (max - min) / effective_size
    bins = jnp.arange(0, effective_size) * step + step / 2
    res = jnp.exp(-(log_sigma[..., None] - bins) ** 2 / (2 * step ** 2))
    return jnp.concatenate((-(log_sigma[..., None] / 4), res), axis=-1)
