import numpy as np

import jax
import jax.numpy as jnp

from salad.aflib.common.protein import Protein
from salad.aflib.common.protein import to_pdb as protein_to_pdb
from salad.aflib.model.all_atom_multimer import atom14_to_atom37, get_atom37_mask

def from_config(config, num_aa=None,
                residue_index=None,
                chain_index=None,
                cyclic_mask=None):
    c = config
    if cyclic_mask is not None:
        c.cyclic = True
    if num_aa is not None:
        if chain_index is None:
            chain_index = jnp.zeros((num_aa,), dtype=jnp.int32)
        if residue_index is None:
            residue_index = jnp.arange(num_aa, dtype=jnp.int32)
    else:
        if chain_index is not None:
            num_aa = chain_index.shape[0]
        if residue_index is not None:
            num_aa = residue_index.shape[0]
    init_pos = jnp.zeros((num_aa, 5 + c.augment_size, 3), dtype=jnp.float32)
    init_aa_gt = jnp.full((num_aa,), 20, dtype=jnp.int32)
    init_local = jnp.zeros((num_aa, c.local_size), dtype=jnp.float32)
    data = dict(
        pos=init_pos,
        aa_gt=init_aa_gt,
        seq=init_aa_gt,
        residue_index=residue_index,
        chain_index=chain_index,
        batch_index=jnp.zeros_like(chain_index),
        mask=jnp.ones_like(chain_index, dtype=jnp.bool_),
        cyclic_mask=cyclic_mask,
        t_pos=jnp.ones((num_aa,), dtype=jnp.float32),
        t_seq=jnp.ones((num_aa,), dtype=jnp.float32)
    )
    prev = dict(
        pos=jnp.zeros_like(init_pos),
        local=init_local
    )
    return data, prev

def update(data, **kwargs):
    data = {k: v for k, v in data.items()}
    for name, item in kwargs.items():
        data[name] = item
    return data

def to_protein(data):
    atom37 = atom14_to_atom37(data["atom_pos"], data["aatype"])
    atom37_mask = get_atom37_mask(data["aatype"])
    residue_index = np.array(data["residue_index"])
    if "result" in data:
        maxprob = jax.nn.softmax(data["result"]["aa"], axis=-1).max(axis=-1)
    else:
        maxprob = np.zeros((residue_index.shape[0],), dtype=np.float32)
    protein = Protein(
        np.array(atom37),
        np.array(data["aatype"]),
        np.array(atom37_mask),
        residue_index,
        np.array(data["chain_index"]),
        maxprob[:, None] * np.array(
            np.ones_like(atom37_mask, dtype=jnp.float32)))
    return protein

def to_pdb(data):
    return protein_to_pdb(to_protein(data))
