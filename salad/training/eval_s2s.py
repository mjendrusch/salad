import time
import os

import pickle

from copy import deepcopy

import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk

from salad.aflib.common.protein import to_pdb, from_pdb_string, Protein
from salad.aflib.model.all_atom_multimer import atom37_to_atom14, atom14_to_atom37, get_atom37_mask
from salad.aflib.model.geometry import Vec3Array

from salad.modules.structure_to_sequence import S2SInference, S2SEfficientInference
from salad.modules.config import structure_to_sequence as config_choices
from salad.data.allpdb import slice_dict
from flexloop.utils import parse_options

def model_step(config):
    module = S2SInference
    if config.decoder_depth:
        module = S2SEfficientInference
    config = deepcopy(config)
    config.eval = True
    def step(data):
        s2s = module(config)
        out = s2s(data)
        return out
    return step

def parse_input_data(path, size=1024):
    file_names = os.listdir(path)
    for name in file_names:
        if not name.endswith(".pdb"):
            continue
        with open(f"{path}/{name}", "rt") as f:
            protein = from_pdb_string(f.read())
        atom_pos_37 = protein.atom_positions
        atom_mask_37 = protein.atom_mask
        aatype = protein.aatype
        resi = protein.residue_index
        chain = protein.chain_index
        batch = jnp.zeros_like(protein.residue_index)
        atom_pos_14, atom_mask_14 = atom37_to_atom14(
            aatype, Vec3Array.from_array(atom_pos_37), atom_mask_37)
        atom_pos_14 = atom_pos_14.to_array()
        data = dict(
            aa_gt=aatype,
            residue_index=resi,
            chain_index=chain,
            batch_index=batch,
            all_atom_positions=atom_pos_14,
            all_atom_mask=atom_mask_14,
            seq_mask=aatype != 20,
            residue_mask=atom_mask_37[:, 1],
            smol_positions=jnp.zeros((batch.shape[0], 16, 3), dtype=jnp.float32),
            smol_types=6 * jnp.ones((batch.shape[0], 16), dtype=jnp.int32),
            smol_mask=jnp.zeros((batch.shape[0], 16), dtype=jnp.bool_),
        )
        data = pad_to_size(data, size=size)
        yield name, data

def pad_to_size(data, size):
    result = dict()
    for key, item in data.items():
        if item.shape[0] < size:
            delta = size - item.shape[0]
            item = np.concatenate((
                item,
                np.zeros([delta] + list(item.shape[1:]), dtype=item.dtype)
            ))
        result[key] = item
    return result

AA_CODE = "ARNDCQEGHILKMFPSTWYVX"
def encode_sequence(sequence):
    return np.array([AA_CODE.index(c) for c in sequence], dtype=np.int32)
def decode_sequence(sequence, mask):
    sequence = np.array(sequence)
    mask = np.array(mask)
    result = []
    for idx in range(len(sequence)):
        if mask[idx]:
            aa = AA_CODE[sequence[idx]]
            result.append(aa)
    return "".join(result)

if __name__ == "__main__":
    opt = parse_options(
        "evaluate a sequence-to-structure model on a set of PDB files.",
        params="checkpoint.jax",
        constraint="none",
        out_path="outputs/",
        config="default",
        path="inputs/",
        diagnostics="False",
        num_recycle=4,
        temperature=0.1,
        replicas=8,
        jax_seed=42
    )

    print(f"Running S2S with {opt.num_recycle + 1} steps and temperature {opt.temperature} on files in {opt.path}")
    start = time.time()

    config = getattr(config_choices, opt.config)
    config.eval = True
    config.num_recycle = opt.num_recycle
    config.temperature = opt.temperature
    key = jax.random.PRNGKey(opt.jax_seed)
    _, model = hk.transform(model_step(config))
    model = jax.jit(model)

    print("Loading model parameters...")
    start = time.time()
    with open(opt.params, "rb") as f:
        params = pickle.load(f)
    print(f"Model parameters loaded in {time.time() - start:.3f} seconds.")

    print("Start decoding...")
    os.makedirs(opt.out_path, exist_ok=True)
    os.makedirs(f"{opt.out_path}/diagnostics/", exist_ok=True)
    with open(f"{opt.out_path}/diagnostics/scores.csv", "wt") as f_scores:
        f_scores.write(f"name,sequence,num_aa,recovery,recovery_null,perplexity,log_p\n")
        for name, data in parse_input_data(opt.path, size=174): # FIXME
            if opt.constraint != "none":
                data["aa_constraint"] = encode_sequence(opt.constraint)
            for replica in range(opt.replicas):
                print(f"Predicting {name}, sequence {replica + 1}...")
                key, subkey = jax.random.split(key)
                mask = data["all_atom_mask"][:, 1] > 0
                out = model(params, subkey, data)
                perplexity = out["perplexity"]
                sequence = out["aatype"]
                recovery = out["recovery"]
                f_scores.write(f"{name},{mask.sum()},{decode_sequence(sequence, mask)},{recovery},{out['recovery_null']},{perplexity},{out['log_p']}\n")
                f_scores.flush()
    print("All proteins decoded.")
