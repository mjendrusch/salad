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

from salad.modules.structure_autoencoder import StructureAutoencoderInference, StructureDecoderInference
from salad.modules.config import distance_to_structure_decoder as config_choices
from salad.data.allpdb import slice_dict
from flexloop.utils import parse_options

def model_step(config):
    module = StructureAutoencoderInference
    if config.is_decoder:
        module = StructureDecoderInference
    config = deepcopy(config)
    config.eval = True
    def step(data):
        decoder = module(config)
        out = decoder(data)
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
            residue_mask=atom_mask_37[:, 1]
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

if __name__ == "__main__":
    opt = parse_options(
        "sample from a protein diffusion model.",
        params="checkpoint.jax",
        out_path="outputs/",
        config="default",
        path="inputs/",
        diagnostics="False",
        time=0.0,
        num_recycle=4,
        jax_seed=42
    )

    print(f"Running decoder with {opt.num_recycle + 1} steps on files in {opt.path}")
    start = time.time()

    config = getattr(config_choices, opt.config)
    config.eval = True
    config.num_recycle = opt.num_recycle
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
        f_scores.write(f"name,num_aa,recovery,perplexity,rmsd_ca,tm,lddt\n")
        for name, data in parse_input_data(opt.path, size=1024):
            data["time"] = opt.time # FIXME: time input
            key, subkey = jax.random.split(key)
            mask = data["all_atom_mask"][:, 1] > 0
            out = model(params, subkey, data)
            out = slice_dict(out, mask)
            atom37 = atom14_to_atom37(out["atom_pos"], out["aatype"])
            atom37_mask = get_atom37_mask(out["aatype"])
            protein = Protein(np.array(atom37), np.array(out["aatype"]),
                            np.array(atom37_mask), np.array(data["residue_index"]),
                            np.array(data["chain_index"]),
                            np.stack([100 * np.array(out["lddt"])] * 37, axis=-1))
            pdb_string = to_pdb(protein)
            mean_lddt = out['lddt'].sum() / data["all_atom_mask"][:, 1].sum()
            f_scores.write(f"{name},{int(mask.astype(jnp.int32).sum())},{float(out['recovery'])},{float(out['perplexity'])},{float(out['rmsd_ca'])},{float(out['tm'])},{float(mean_lddt)}\n")
            f_scores.flush()
            with open(f"{opt.out_path}/decoder_{name}", "wt") as f:
                f.write(pdb_string)
            if opt.diagnostics == "True":
                diagnostics = dict(
                    latent=np.array(out["latent"]),
                    local=np.array(out["local"]),
                    dssp=np.array(out["dssp"])
                )
                if "codebook_index" in out:
                    diagnostics["codebook_index"] = np.array(out["codebook_index"])
                np.savez_compressed(
                    f"{opt.out_path}/diagnostics/decoder_{'.'.join(name.split('.')[:-1])}.npz",
                    **diagnostics
                )
    print("All proteins decoded.")
