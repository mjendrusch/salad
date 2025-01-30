import jax
import jax.numpy as jnp

from alphafold.model.geometry import Vec3Array

from salad.modules.utils.geometry import index_sum, extract_aa_frames, distance_one_hot

def get_residue_nll(logits, ground_truth, predict_mask, batch, mask,
                average="per_item", num_classes=20):
    if average == "per_item":
        weight = 1 / jnp.maximum(index_sum(predict_mask, batch, mask), 1) / (batch.max() + 1)
    elif average == "per_residue":
        weight = 1 / jnp.maximum(predict_mask.sum(keepdims=True), 1)
    else:
        raise ValueError(f"'average' must be one of 'per_item' or 'per_residue'"
                         f" not {average}!")
    weight = jnp.where(predict_mask, weight, 0)
    print(weight.shape)
    nll = -(logits * jax.nn.one_hot(ground_truth, num_classes, axis=-1)).sum(axis=-1)
    print(nll.shape)
    nll = jnp.where(predict_mask, nll, 0)
    print(nll.shape)
    nll = (nll * weight).sum()
    print(nll.shape)
    return nll

def get_batch_weight(batch: jnp.ndarray,
                     mask: jnp.ndarray) -> jnp.ndarray:
    return mask / jnp.maximum(
        index_sum(
            mask.astype(jnp.float32),
            batch, mask), 1) / (batch.max() + 1)

def get_rotation_loss(pos, pos_gt, weight) -> jnp.ndarray:
    # if we're computing the loss for a trajectory
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
    rotation_product = (gt_frames.rotation.inverse() @ frames.rotation).to_array()
    rotation_loss = ((rotation_product - jnp.eye(rotation_product.shape[-1])) ** 2).sum(axis=(-1, -2))
    rotation_loss = (rotation_loss * weight).sum(axis=-1)
    if is_trajectory:
        rotation_loss = rotation_loss.mean()
    return rotation_loss

def get_distogram_nll(logits, pos_gt, batch_index, mask, atom_index=1,
                      min_distance=0.0, max_distance=22.0,
                      bins=64):
    if not isinstance(pos_gt, Vec3Array):
        pos_gt = Vec3Array.from_array(pos_gt)
    pos_x = pos_gt[:, atom_index]
    pair_mask = mask[:, None] * mask[None, :]
    pair_mask *= batch_index[:, None] == batch_index[None, :]
    distogram = distance_one_hot((pos_x[:, None] - pos_x[None, :]).norm(),
                                 min_distance=min_distance,
                                 max_distance=max_distance,
                                 bins=bins)
    nll = -(logits * distogram).sum(axis=-1)
    nll = jnp.where(pair_mask, nll, 0).sum()
    nll /= jnp.maximum(pair_mask.sum(), 1)
    return nll
