"""This module contains utilities for applying diffusion noise to different data types."""

from typing import Optional, Union
import jax
import jax.numpy as jnp
from salad.aflib.model.geometry import Vec3Array, Rigid3Array, Rot3Array

from salad.modules.utils.geometry import index_mean, index_count

Float = Union[float, jnp.ndarray]

def diffuse_sequence(key, x: jnp.ndarray, t: Float, mask_index=20):
    """Randomly mask an input sequence.
    
    Args:
        key: jax RNG key.
        x: discrete sequence data.
        t: diffusion time. Corresponds to % of positions masked.
        mask_index: token index used for masking. Default: 20.
    Returns:
        Masked sequence and boolean array used for masking.
    """
    corrupt_aa = jax.random.bernoulli(key, p=t, shape=x.shape)
    out = jnp.where(corrupt_aa, mask_index, x)
    return out, corrupt_aa

def diffuse_coordinates_edm(key, x: Vec3Array, batch_index,
                            t: jnp.ndarray, symm=None) -> Vec3Array:
    """Apply variance expanding noise to an input structure.
    
    Args:
        key: jax RNG key.
        x: input structure coordinates.
        batch_index: batch index.
        t: diffusion time / standard deviation of noise.
        symm: optional symmetrizer applied to the noise.
    Returns:
        Noisy structure.
    """
    target = jax.random.normal(key, [x.shape[0], x.shape[1], 3])
    center = jnp.zeros_like(target)[:, 1].at[batch_index].add(target[:, 1])
    center /= jnp.maximum(jnp.zeros_like(target)[:, 1].at[batch_index].add(1.0), 1e-6)
    center = center[batch_index]
    target = target - center[:, None, :]
    if symm is not None:
        target = symm(target)
    target = Vec3Array.from_array(target)
    target = t * target
    result = x + target
    return result

def position_std(x, index, mask):
    """Compute the standard deviation of CA positions of an input structure.
    
    Args:
        x: structure coordinates.
        index: integer index for grouping chains or items in a batch.
        mask: residue mask.
    Returns:
        Broadcasted standard deviation of CA atom positions.
    """
    ca = x[:, 1]
    center = index_mean(ca, index, mask[..., None])
    ca -= center
    return jnp.sqrt(
        index_mean(ca ** 2, index, mask[..., None]).mean(axis=1)
      - index_mean(ca, index, mask[..., None]).mean(axis=1) ** 2)

def diffuse_atom_cloud(key, x: Vec3Array, mask: jnp.ndarray, batch_index: jnp.ndarray,
                       t: jnp.ndarray, cloud_std=None, flow=False,
                       residue_std=4.0, correlated_residues=False,
                       symm=None) -> Vec3Array:
    """Interpolate between a random point cloud and an input structure.
    
    Args:
        key: jax RNG key.
        x: structure coordinates.
        mask: residue mask.
        batch_index: batch index.
        t: diffusion time.
        cloud_std: random point cloud standard deviation.
        flow: use flow-matching interpolation (linear). Default: False.
        residue_std: standard deviation of residue atoms if using correlated_residues.
        correlated_residues: use correlated noise, sampling residue centers first,
            then sampling residue atoms around them. Default: False.
        symm: optional symmetrizer function.
    Returns:
        Noisy atom point cloud.
    """
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

def diffuse_features(key, x: jnp.ndarray, t: float, scale=4.0) -> jnp.ndarray:
    """Interpolate between a feature vector and random noise.
    
    Args:
        x: input feature vector.
        t: diffusion time.
        scale: noise scale. Default: 4.0.
    Returns:
        Noisy feature vector.
    """
    target = jax.random.normal(key, x.shape) * scale
    return jnp.sqrt(t) * target + jnp.sqrt(1 - t) * x

def diffuse_features_edm(key, x: jnp.ndarray, t: float) -> jnp.ndarray:
    """Add variance-expanding noise to a feature vector.
    
    Args:
        x: input feature vector.
        t: diffusion time / noise standard deviation.
    Returns:
        Noisy feature vector.
    """
    target = jax.random.normal(key, x.shape)
    result = x + t * target
    return result

def diffuse_features_vp(key, x: jnp.ndarray, t: float, scale=1.0) -> jnp.ndarray:
    """Interpolate between a feature vector and random noise with VP noise scaling.
    
    Args:
        x: input feature vector.
        t: diffusion time.
        scale: noise scale. Default: 1.0.
    Returns:
        Noisy feature vector.
    """
    target = jax.random.normal(key, x.shape) * scale
    s = 0.01
    alpha_bar = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
    alpha = jnp.sqrt(alpha_bar)
    sigma = jnp.sqrt(1 - alpha_bar)
    result = x * alpha + sigma * target
    return result

def time_embedding(t, size=128, start=0, stop=1) -> jnp.ndarray:
    """Binned time embedding.
    
    Args:
        t: diffusion time.
        size: number of bins.
        start: start of first bin.
        stop: end of last bin.
    Returns:
        RBF embedding of diffusion time.
    """
    bins = jnp.arange(size) / size + 1 / (2 * size)
    bins = bins * (stop - start) + start
    return jnp.exp(-size * (t[..., None] - bins) ** 2)

def fourier_time_embedding(t, size=128) -> jnp.ndarray:
    """Fourier time embedding.
    
    Args:
        t: diffusion time.
        size: number of fourier features.
    Returns:
        Fourier time embedding.
    """
    exponent = 2 * jnp.arange(0, size // 2, dtype=t.dtype) / size
    denominator = 10_000 ** exponent
    result = (t[..., None]) / denominator
    return jnp.concatenate((jnp.sin(result), jnp.cos(result)), axis=-1)

def log_sigma_embedding(log_sigma, size=128, min=-4, max=0) -> jnp.ndarray:
    """RBF embedding of the log standard deviation.

    Args:
        log_sigma: log standard deviation.
        size: number of bins.
        min: start of the first bin.
        max: end of the last bin.
    Returns:
        RBF embedding of the log standard deviation. 
    """
    effective_size = size - 1
    step = (max - min) / effective_size
    bins = jnp.arange(0, effective_size) * step + step / 2
    res = jnp.exp(-(log_sigma[..., None] - bins) ** 2 / (2 * step ** 2))
    return jnp.concatenate((-(log_sigma[..., None] / 4), res), axis=-1)

def diffuse_features_permute(key, x: jnp.ndarray, t: float, mask: jnp.ndarray, scale=1.0) -> jnp.ndarray:
    noise_key, permute_key, choose_permuted_key = jax.random.split(key, num = 3)
    target = jax.random.normal(noise_key, x.shape) * scale
    x_permute = jax.lax.stop_gradient(x)
    x_permute = jnp.where(mask[..., None], x_permute, target)
    permuted_x = jax.random.permutation(permute_key, x_permute, axis=0, independent=False)
    choose_permuted = jax.random.bernoulli(choose_permuted_key, 0.15, (x_permute.shape[0],))
    target = jnp.where(choose_permuted[..., None], permuted_x, target)
    sigma = t
    alpha = jnp.sqrt(1 - t ** 2)
    result = x * alpha + sigma * target
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
