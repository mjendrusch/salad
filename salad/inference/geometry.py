import numpy as np
import jax
import jax.numpy as jnp
from salad.modules.utils.geometry import index_align, index_mean, positions_to_ncacocb

def center_positions(pos, index=None, mask=None):
    """Centers positions on the center of mass of anchor atoms (CA).
    
    Args:
        pos: positions array of shape (N, M, 3), with anchor atoms at index [:, 1].
        index: batch index for multiple centering.
        mask: mask of occupied positions.

    Returns:
        centered positions: positions centered on the CA center of mass.
    """
    if index is None:
        index = jnp.zeros((pos.shape[0],), dtype=jnp.int32)
    if mask is None:
        mask = jnp.ones((pos.shape[0],), dtype=jnp.int32)
    return pos - index_mean(pos[:, 1], index, mask[:, None])[:, None]

def motif_to_pseudo(motif, pos):
    ncacocb = positions_to_ncacocb(motif)
    return jnp.concatenate((ncacocb, jnp.repeat(ncacocb[4:5], pos.shape[1] - 5)), axis=1)

def align_set_positions(motif, pos, has_motif, index=None, mask=None,
                        replace_pseudo=True):
    """Aligns a motif to a position array and replaces aligned positions
    with motif positions.

    Args:
        motif: motif position array of shape (N, M, 3).
        pos: position array of shape (N, M, 3)
        has_motif: motif mask of shape (N,)
        index: batch index for multiple alignments.
        mask: mask of occupied positions.
        replace_pseudo: align and replace pseudoatoms from the motif?
            If False, only backbone atoms (N through CB) will be replaced.
            Default: True.

    Returns:
        motif-replaced positions: positions where all entries with has_motif = True
            have been replaced by aligned motif positions.
    """
    if index is None:
        index = jnp.zeros((pos.shape[0],), dtype=jnp.int32)
    if mask is None:
        mask = jnp.ones((pos.shape[0],), dtype=jnp.int32)
    align_mask = mask
    align_weight = jnp.array(has_motif, dtype=jnp.float32)
    update_pos = index_align(
        motif, pos,
        index=index,
        mask=align_mask,
        weight=align_weight)
    if not replace_pseudo:
        update_pos = update_pos.at[5:].set(pos.at[5:])
    return jnp.where(has_motif, update_pos, pos)
