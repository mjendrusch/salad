
import jax
import jax.numpy as jnp
import haiku as hk

from salad.modules.basic import Linear
from salad.modules.geometric import distance_rbf
from salad.aflib.model.geometry import Vec3Array

# TODO
class NonEquivariantAtomMotifEmbedding(hk.Module):
    def __init__(self, config, name = "tip_atom_embedding"):
        super().__init__(name)
        self.config = config

    def __call__(self, aatype, pos, floating, group, resi, chain, batch, atom_mask):
        c = self.config
        local_embedding = ... # TODO
        pair_embedding = ... # TODO
        mean_pos = (atom_mask[..., None] * pos).sum(axis=1) / jnp.maximum(atom_mask.sum(axis=1), 1)
        compare = Vec3Array.from_array(mean_pos)
        mean_pos_pair = (compare[:, None] - compare[None, :]).norm()
        mean_pos_pair = Linear(c.pair_size)(distance_rbf(mean_pos_pair, 0.0, 0.0, 16))
        rel_pos = (pos[None, :] - mean_pos[:, None])
        pos = jnp.where(atom_mask[..., None], pos, mean_pos)

        pos = Vec3Array.from_array(pos)
        pos.cross()
        dist = (pos[:, None, :, None] - pos[None, :, None, :]).norm() # N * N * 14 * 14 = N**2 * 196
        return local_embedding, pair_embedding

