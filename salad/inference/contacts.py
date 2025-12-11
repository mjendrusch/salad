import jax
import jax.numpy as jnp

from salad.modules.utils.geometry import index_mean

def compact_step(positions, index, mask):
    ca = positions[:, 1]
    update = index_mean(ca, index, mask[:, None]) - ca
    return update[:, None, :]

def clash_step(positions, residue_index, chain_index, batch_index=None, threshold=8.0):
    same_chain = chain_index[:, None] == chain_index[None, :]
    rdist = jnp.where(
        same_chain,
        abs(residue_index[:, None] - residue_index[None, :]),
        jnp.inf)
    long_range = rdist > 16
    if batch_index is not None:
        same_batch = batch_index[:, None] == batch_index[None, :]
        long_range *= same_batch
    def clashy(x):
        clash_threshold = threshold
        dist = jnp.sqrt(jnp.maximum(
            ((x[:, None] - x[None, :]) ** 2).sum(axis=-1), 1e-6))
        clashyness = jnp.where(
            long_range,
            jax.nn.relu(clash_threshold - dist) / clash_threshold, 0).sum()
        return clashyness
    return -jax.grad(clashy, argnums=(0,))(positions[:, 1])[0][:, None]

def contact_step(x, contact_mask, d0=4.0, r0=8.0, eps=1e-6):
    def contact_loss(ca):
        rij = jnp.sqrt(jnp.maximum((ca[:, None] - ca[None, :]) ** 2, eps).sum(axis=-1))
        sij = (1 - ((rij - d0) / r0) ** 6 + eps) / (1 - ((rij - d0) / r0) ** 12 + eps)
        return (sij * contact_mask).mean()
    return jax.grad(contact_loss, argnums=(0,))(x)[0][:, None]
