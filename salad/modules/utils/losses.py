r"""This module contains common loss functions for training
models on protein sequence and structure data."""

import jax
import jax.numpy as jnp

from salad.aflib.model.geometry import Vec3Array

from salad.modules.utils.geometry import index_sum, extract_aa_frames, distance_one_hot

def get_residue_nll(
        logits, ground_truth, predict_mask, batch_index, mask,
        average="per_item", num_classes=20):
    r"""Compute and average residue identity negative log likelihood loss.
    
    Args:
        logits: normalized logits of shape (..., num_classes).
        ground_truth: ground truth labels or probability distribution of shape (..., num_classes),
        predict_mask: mask of residues to compute losses for.
        batch_index: unique index of each item in a batch.
        mask: general residue mask.
        average: type of averaging applied. One of
            "per_item" (average per item in the batch_index) or
            "per_residue" (average over all residues)
    Returns:
        Averaged negative log likelihood loss.
    """
    if average == "per_item":
        weight = 1 / jnp.maximum(index_sum(
            predict_mask, batch_index, mask), 1) / (batch_index.max() + 1)
    elif average == "per_residue":
        weight = 1 / jnp.maximum(predict_mask.sum(keepdims=True), 1)
    else:
        raise ValueError(f"'average' must be one of 'per_item' or 'per_residue'"
                         f" not {average}!")
    weight = jnp.where(predict_mask, weight, 0)
    nll = -(logits * jax.nn.one_hot(ground_truth, num_classes, axis=-1)).sum(axis=-1)
    nll = jnp.where(predict_mask, nll, 0)
    nll = (nll * weight).sum()
    return nll

def get_batch_weight(batch_index: jnp.ndarray,
                     mask: jnp.ndarray) -> jnp.ndarray:
    """Compute a per residue loss weight that corresponds to a per-item average.
    
    Args:
        batch_index: unique index of each item in a batch.
        mask: residue mask.
    Returns:
        Per-residue loss weights of shape (N,).
    """
    return mask / jnp.maximum(
        index_sum(
            mask.astype(jnp.float32),
            batch_index, mask), 1) / (batch_index.max() + 1)

def get_rotation_loss(pos, pos_gt, weight) -> jnp.ndarray:
    """Compute rotation-similarity loss between predicted positions
       pos and ground truth positions pos_gt.
    
    Args:
        pos: predicted positions in order (N, CA, C, ...) of shape (..., N, 3+, 3).
        pos_gt: ground truth positions in order (N, CA, C, ...) of shape (..., N, 3+, 3).
        weight: per residue loss weight.
    Returns:
        Averaged rotation similarity loss.
    """
    # check, if we're computing the loss for a trajectory
    is_trajectory = False
    extract_frames = extract_aa_frames
    if len(pos.shape) == len(pos_gt.shape) + 1:
        pos_gt = pos_gt[None]
        is_trajectory = True
        extract_frames = jax.vmap(extract_frames)
    gt_frames, _ = extract_frames(
        Vec3Array.from_array(pos_gt))
    frames, _ = extract_frames(
        Vec3Array.from_array(pos))
    # compute the relative rotation between the same frame in
    # pos and pos_gt
    rotation_product = (gt_frames.rotation.inverse() @ frames.rotation).to_array()
    # constrain the relative rotation to be close to the identity matrix
    rotation_loss = ((rotation_product - jnp.eye(rotation_product.shape[-1])) ** 2).sum(axis=(-1, -2))
    rotation_loss = (rotation_loss * weight).sum(axis=-1)
    # if the input was a trajectory, average the loss over the trajectory
    if is_trajectory:
        rotation_loss = rotation_loss.mean()
    return rotation_loss

def get_distogram_nll(logits, pos_gt, batch_index, mask, atom_index=1,
                      min_distance=0.0, max_distance=22.0,
                      bins=64):
    """Compute a distogram negative log likelihood loss.

    Args:
        logits: normalized distogram logits of shape (N, N, bins)
        pos_gt: atom14 format ground truth positions of shape (N, 4+, 3)
        batch_index: unique index of each item in a batch.
        mask: residue mask.
        atom_index: index into axis 1 of pos_gt selecting the atoms used
            for distance calculation. Default = 1, corresponding to CA.
        min_distance: minimum distance in angstroms for distance binning. Default: 0.0.
        max_distance: maximum distance in angstroms for distance binning. Default: 22.0.
        bins: number of distogram bins.
    Returns:
        Distogram negative log likelihood averaged over all residue pairs.
    """
    # convert pos_gt to Vec3Array for convenience
    if not isinstance(pos_gt, Vec3Array):
        pos_gt = Vec3Array.from_array(pos_gt)
    pos_x = pos_gt[:, atom_index]
    pair_mask = mask[:, None] * mask[None, :]
    pair_mask *= batch_index[:, None] == batch_index[None, :]
    # compute the ground-truth distogram from the selected
    # atom positions
    distogram = distance_one_hot((pos_x[:, None] - pos_x[None, :]).norm(),
                                 min_distance=min_distance,
                                 max_distance=max_distance,
                                 bins=bins)
    # compute negative log likelihood.
    nll = -(logits * distogram).sum(axis=-1)
    nll = jnp.where(pair_mask, nll, 0).sum()
    nll /= jnp.maximum(pair_mask.sum(), 1)
    return nll
