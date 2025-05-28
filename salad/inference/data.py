from copy import copy

import numpy as np

import jax
import jax.numpy as jnp

from salad.aflib.common.protein import Protein
from salad.aflib.common.protein import to_pdb as protein_to_pdb
from salad.aflib.model.all_atom_multimer import atom14_to_atom37, get_atom37_mask

def from_config(config, num_aa=None,
                residue_index=None,
                chain_index=None,
                cyclic_mask=None,
                padded_to=None):
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
    if padded_to is not None:
        data = pad(data, size=padded_to)
        prev = pad(prev, size=padded_to)
    return data, prev

def apply(x: dict, function):
    return {k: function(v) for k, v in x.items()}

def pad(x: dict, size: int):
    return {
        k: np.concatenate((
            v, np.zeros([size - v.shape[0]] + list(v.shape[1:]), dtype=v.dtype)), axis=0)
        for k, v in x.items()
    }

def repeat(data: dict, prev: dict | None = None, count=2, join_chains=()):
    # make the requisite number of copies
    repeated = []
    for idx in range(count):
        subunit = copy(data)
        subunit["repeat"] = np.full_like(data["residue_index"], idx)
        repeated.append(subunit)
    repeated = concatenate(repeated)
    # rebind chains and residue indices
    join_resi = dict()
    join_chain = 0
    chain_mapping = dict()
    new_resi = []
    new_chain = []
    for rep, exists, resi, chain in zip(repeated["repeat"],
                                        np.array(repeated["mask"]),
                                        np.array(repeated["residue_index"]),
                                        np.array(repeated["chain_index"])):
        if chain in join_chains:
            if chain not in join_resi:
                join_resi[chain] = 0
                chain_mapping[chain] = join_chain
                join_chain += 1
            new_resi.append(join_resi[chain])
            new_chain.append(chain_mapping[chain])
            if exists:
                join_resi[chain] += 1
        else:
            if (chain, rep) not in chain_mapping:
                chain_mapping[chain, rep] = join_chain
                join_chain += 1
            new_resi.append(resi)
            new_chain.append(chain_mapping[chain, rep])
    repeated["residue_index"] = jnp.array(new_resi, dtype=jnp.int32)
    repeated["chain_index"] = jnp.array(new_chain, dtype=jnp.int32)
    # remove the now-redundant repeat index
    del repeated["repeat"]
    # optionally repeat and return prev data as well
    if prev is not None:
        repeated_prev = concatenate(count * [prev])
        return repeated, repeated_prev
    return repeated

def concatenate(data_list):
    return {
        k: np.concatenate([item[k] for item in data_list], axis=0)
        for k in data_list[0]
    }

def select(data, index):
    return {k: v[index] for k, v in data.items()}

def filter(data, expr):
    return {k: v for k, v in data.items() if expr(k)}

def drop_masked(data):
    return select(data, data["mask"] > 0)

def reorder_chains(data, chain_index=None):
    if chain_index is None:
        chain_index = data["chain_index"]
    unique_chains = np.unique(chain_index)
    results = []
    for c in unique_chains:
        results.append(select(data, chain_index == c))
    return concatenate(results)

def update(data, **kwargs):
    data = {k: v for k, v in data.items()}
    for name, item in kwargs.items():
        data[name] = item
    return data

def to_protein(data):
    atom37 = atom14_to_atom37(data["atom_pos"], data["aatype"])
    atom37_mask = get_atom37_mask(data["aatype"])
    residue_index = np.array(data["residue_index"])
    if "aa_logits" in data:
        maxprob = jax.nn.softmax(data["aa_logits"], axis=-1).max(axis=-1)
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
