import time
import os

import pickle

from copy import deepcopy

import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk

from alphafold.common.protein import to_pdb, from_pdb_string, Protein
from alphafold.model.all_atom_multimer import (
    atom37_to_atom14, atom14_to_atom37, get_atom14_mask, get_atom37_mask)
from alphafold.model.geometry import Vec3Array

from salad.modules.structure_autoencoder import StructureAutoencoderInference, StructureDecoderInference
from salad.modules.config import distance_to_structure_decoder as config_choices
from salad.data.allpdb import slice_dict
from salad.training.eval_noise_schedule_benchmark import parse_num_aa
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
        num_aa="100",
        steps=50,
        num_designs=10,
        diagnostics="False",
        latent_diffusion="False",
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

    num_aa, resi, chain, is_cyclic, cyclic_mask = parse_num_aa(opt.num_aa)
    batch = jnp.zeros_like(resi)

    print("Loading model parameters...")
    start = time.time()
    with open(opt.params, "rb") as f:
        params = pickle.load(f)
    print(f"Model parameters loaded in {time.time() - start:.3f} seconds.")

    print("Start decoding...")
    os.makedirs(opt.out_path, exist_ok=True)
    os.makedirs(f"{opt.out_path}/diagnostics/", exist_ok=True)
    for idx in range(opt.num_designs):
        data = dict(
            aa_gt=jnp.zeros_like(resi),
            residue_index=resi,
            chain_index=chain,
            batch_index=batch,
            all_atom_positions=10 * jax.random.normal(key, (resi.shape[0], 14, 3), dtype=jnp.float32),
            all_atom_mask=jnp.ones((resi.shape[0], 14), dtype=jnp.bool_),
            seq_mask=jnp.ones_like(resi, dtype=jnp.bool_),
            residue_mask=jnp.ones_like(resi, dtype=jnp.bool_),
            latent=0 * jax.random.normal(key, (resi.shape[0], config.local_size if config.noembed else config.latent_size), dtype=jnp.float32)
        )
        for step in range(opt.steps):
            key, subkey = jax.random.split(key)
            t = 1 - step / opt.steps
            # t = step / (opt.steps - 1)
            # t = (80.0 ** (1 / 7) + t * (0.01 ** (1 / 7) - 80.0 ** (1 / 7))) ** 7
            data["time"] = t
            out = model(params, subkey, data)
            aatype = out["aatype"]
            print(step, float(out["time"][0]), float(out["violation"]))
            data["aa_gt"] = aatype
            data["all_atom_positions"] = 1.0 * out["atom_pos"]
            data["all_atom_mask"] = get_atom14_mask(aatype)
            if opt.latent_diffusion == "True":
                data["latent"] = out["latent"]
            if step % 5 == 0:
                atom37 = atom14_to_atom37(out["atom_pos"], out["aatype"])
                atom37_mask = get_atom37_mask(out["aatype"])
                protein = Protein(np.array(atom37), np.array(out["aatype"]),
                                np.array(atom37_mask), np.array(data["residue_index"]),
                                np.array(data["chain_index"]),
                                np.stack([100 * np.array(out["lddt"])] * 37, axis=-1))
                pdb_string = to_pdb(protein)
                mean_lddt = out['lddt'].sum() / data["all_atom_mask"][:, 1].sum()
                with open(f"{opt.out_path}/design_{idx}_{step}.pdb", "wt") as f:
                    f.write(pdb_string)
    print("All proteins generated.")
