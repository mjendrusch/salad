import jax
import jax.numpy as jnp
import haiku as hk

class PairProcessingBlock(hk.Module):
    def __init__(self, config, name = None):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, pos, condition, index, mask):
        c = self.config
        
        pass # TODO

class SparseProcessingBlock(hk.Module):
    def __init__(self, config, name = None):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, pos, condition, index, mask):
        c = self.config

        pass # TODO

class DenoisingBlock(hk.Module):
    def __init__(self, config, name = None):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, pos, condition, index, mask):
        c = self.config
        neighbours = get_neighbours_v2(pos, index, mask)
        pair_features = sparse_pair_features_v2(pos, pair, neighbours)
        local += hk.remat(GatedHybridAttention(c))(
            hk.LayerNorm([-1], True, True)(local),
            pair_features, condition, pos, mask)
        local += GatedTransition(c)(
            hk.LayerNorm([-1], True, True)(local, condition))
        if c.num_local_blocks:
            for i in range(c.num_local_blocks):
                local += hk.remat(GatedAttention(c))(local, index, mask)
                local += GatedTransition(c)(
                    hk.LayerNorm([-1], True, True)(local, condition))
        pos += PositionUpdate(c)(local)
        return local, pos

