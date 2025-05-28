import jax
import jax.numpy as jnp

from salad.modules.utils.geometry import index_mean

def compact_step(positions, index):
    ca = positions[:, 1]
    update = index_mean(ca, index) - ca
    return update[:, None, :]

def clash_step(positions, residue_index, chain_index, threshold=8.0):
    same_chain = chain_index[:, None] == chain_index[None, :]
    rdist = jnp.where(
        same_chain,
        abs(residue_index[:, None] - residue_index[None, :]),
        jnp.inf)
    long_range = rdist > 16
    def clashy(x):
        clash_threshold = threshold
        dist = jnp.sqrt(jnp.maximum(
            ((x[:, None] - x[None, :]) ** 2).sum(axis=-1), 1e-6))
        clashyness = jnp.where(
            long_range,
            jax.nn.relu(clash_threshold - dist) / clash_threshold, 0).sum()
        return clashyness
    return -jax.grad(clashy, argnums=(0,))(positions[:, 1])[0][:, None]
