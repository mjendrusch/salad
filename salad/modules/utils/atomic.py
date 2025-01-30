import jax
import jax.numpy as jnp

def make_bondmap(atom_index, bond_index, bond_type, n, c, o3, p,
                 residue_index, chain_index, batch_index):
    same_batch = batch_index[:, None] * batch_index[None, :]
    same_chain = chain_index[:, None] * chain_index[None, :]
    same_residue = residue_index[:, None] == residue_index[None, :]
    # within-residue bonds
    bondmap = ((bond_index[:, None, :] == atom_index[None, :, None]) * bond_type[:, None, :]).sum(axis=-1)
    bondmap *= same_chain * same_batch * same_residue
    # interresidue bonds
    next_residue = residue_index[:, None] == residue_index[None, :] + 1
    c_to_n = next_residue * c[:, None] * n[:, None]
    n_to_c = c_to_n.T
    o3_to_p = next_residue * o3[:, None] * p[:, None]
    p_to_o3 = o3_to_p.T
    bondmap += c_to_n + n_to_c + o3_to_p + p_to_o3
    return jnp.where(same_batch * same_chain, bondmap, 0)

def make_bond_distance(bondmap, max_dist=7):
    is_bonded = bondmap > 0
    accessibility = 2 * jnp.eye(
        is_bonded.shape[0], dtype=jnp.int32)
    def body(i, acc):
        acc = acc + (acc > 0) @ is_bonded
        return acc
    accessibility = jax.lax.fori_loop(0, max_dist, body, accessibility)
    bond_dist = max_dist + 1 - accessibility
    return bond_dist

def bond_neighbours(bondmap, max_bonds=12):
    is_bonded = bondmap > 0
    index = jnp.arange(is_bonded.shape[0], dtype=jnp.int32)
    neighbours = jnp.argsort(-is_bonded, axis=1)[:max_bonds]
    neighbours = jnp.where(
        is_bonded[index[:, None], neighbours] > 0,
        neighbours, -1)
    return neighbours
