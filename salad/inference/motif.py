
import jax
import jax.numpy as jnp

from salad.modules.utils.geometry import index_align, compute_pseudo_cb

class MotifAdjust:
    def update_pos(self, pos, motifs):
        if isinstance(motifs, dict):
            assert "motif_group" in motifs
            motif = motifs["motif_pos"]
            mask = motifs["has_motif"]
            group = motifs["motif_group"]
            motif_at_pos = index_align(
                motif, pos, group, mask)
            pos = pos.at[:, :5].set(
                jnp.where(mask[:, None, None], motif_at_pos, pos[:, :5]))
            pos = pos.at[:, 5:].set(
                jnp.where(mask[:, None, None], motif_at_pos[:, 1:2], pos[:, 5:]))
        else:
            for motif, mask in motifs:
                motif_at_pos = index_align(
                    motif, pos,
                    jnp.zeros((pos.shape[0],), dtype=jnp.int32),
                    mask)
                pos = pos.at[:, :5].set(
                    jnp.where(mask[:, None, None], motif_at_pos, pos[:, :5]))
                pos = pos.at[:, 5:].set(
                    jnp.where(mask[:, None, None], motif_at_pos[:, 1:2], pos[:, 5:]))
        return pos

def make_dmap(data):
    cb = compute_pseudo_cb(data["motif_pos"])
    has_motif = data["has_motif"]
    has_motif = has_motif[:, None] * has_motif[None, :]
    group = data["motif_group"]
    same_group = group[:, None] == group[None, :]
    dmap = jnp.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
    dmap_mask = has_motif * same_group > 0
    dmap = jnp.where(dmap_mask, dmap, 0)
    return dict(dmap=dmap, dmap_mask=dmap_mask)
