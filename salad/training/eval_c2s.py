import time
import os
from typing import Callable
import random

import pickle

from copy import deepcopy

import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk

from salad.aflib.common.protein import to_pdb, Protein, from_pdb_string
from salad.aflib.model.geometry import Vec3Array
from salad.aflib.model.all_atom_multimer import atom14_to_atom37, atom37_to_atom14, get_atom37_mask

from salad.modules.condition_to_sequence import (
    C2SInference)
from salad.modules.config import condition_to_sequence as config_choices
from flexloop.utils import parse_options

def model_step(config):
    module = C2SInference
    config = deepcopy(config)
    config.eval = True
    def step(data):
        model = module(config)
        out = model(data)
        return out
    return step

def parse_num_aa(data):
    data = data.strip()
    sizes = data.split(":")
    cyclic_mask = []
    resi = []
    chain = []
    num_aa = 0
    is_cyclic = False
    for idx, size in enumerate(sizes):
        if size.startswith("c"):
            size = int(size[1:])
            cyclic = [True]
            is_cyclic = True
        else:
            size = int(size)
            cyclic = [False]
        last_resi = resi[-1] + 50 if resi else 0
        resi += [last_resi + i for i in range(size)]
        chain += size * [idx]
        cyclic_mask += size * cyclic
        num_aa += size
    resi = jnp.array(resi)
    chain = jnp.array(chain)
    cyclic_mask = jnp.array(cyclic_mask)
    return num_aa, resi, chain, is_cyclic, cyclic_mask

def random_dssp_mean(max_loop=0.5):
    helix = np.random.random()
    sheet = 1 - helix
    loop = np.random.random() * max_loop
    return np.array([loop, (1 - loop) * helix, (1 - loop) * sheet])

def random_dssp(count, p=0.8):
    loop, helix, sheet = random_dssp_mean()
    helix = int(helix * count)
    if helix < 6:
        helix = 0
    min_helix_count = 1
    max_helix_count = helix // 6
    min_helix_count = min(min_helix_count, max_helix_count)
    if helix == 0:
        num_helices = 0
    elif min_helix_count == max_helix_count:
        num_helices = min_helix_count
    else:
        num_helices = np.random.randint(min_helix_count, max_helix_count)
    sheet = int(sheet * count)
    if sheet < 8:
        sheet = 0
    loop = count - (helix + sheet)
    min_sheet_count = 2
    max_sheet_count = sheet // 4
    if sheet == 0:
        num_sheets = 0
    elif min_sheet_count == max_sheet_count:
        num_sheets = min_sheet_count
    else:
        num_sheets = np.random.randint(min_sheet_count, max_sheet_count)
    helices = [6 for _ in range(num_helices)] 
    sheets = [4 for _ in range(num_sheets)]
    loops = [0 for _ in range(num_helices + num_sheets + 1)]
    while sum(sheets) < sheet:
        index = np.random.randint(num_sheets)
        sheets[index] += 1
    while sum(helices) < helix:
        index = np.random.randint(num_helices)
        helices[index] += 1
    while sum(loops) < loop:
        index = np.random.randint(0, len(loops))
        loops[index] += 1
    helices = ["_" + "H" * (num - 2) + "_" for num in helices]
    sheets = ["_" + "E" * (num - 2) + "_" for num in sheets]
    loops = ["_" * num for num in loops]
    structured = helices + sheets
    random.shuffle(structured)
    dssp = loops[0] + "".join([s + l for s, l in zip(structured, loops[1:])])
    print(dssp)
    dssp = parse_dssp(dssp)
    return dssp

def parse_dssp_mean(data):
    if data == "none":
        return None
    if data == "random":
        return random_dssp_mean
    return np.array([
        float(c.strip())
        for c in data.strip().split(",")])

def parse_dssp(data):
    DSSP_CODE = "LHE_"
    if data == "none":
        return None
    if data == "random":
        return random_dssp
    return np.array([DSSP_CODE.index(c) for c in data.strip()], dtype=np.int32)

def decode_sequence(x: np.ndarray) -> str:
    AA_CODE = "ARNDCQEGHILKMFPSTWYVX"
    x = np.array(x)
    return "".join([AA_CODE[c] for c in x])

if __name__ == "__main__":
    opt = parse_options(
        "sample from a protein diffusion model.",
        params="checkpoint.jax",
        out_path="outputs/",
        config="default",
        timescale_pos="cosine(t)",
        timescale_seq="1",
        num_aa="100",
        sym="False",
        merge_chains="False",
        replicate_count=3,
        screw_angle=30.0,
        screw_translation=15.0,
        screw_radius=0.0,
        sym_mean="True",
        num_designs=10,
        num_steps=500,
        out_steps="499",
        dssp_mean="none",
        dssp="random",
        temperature=0.1,
        jax_seed=42
    )
    dssp_mean = parse_dssp_mean(opt.dssp_mean)
    dssp = parse_dssp(opt.dssp)

    print(f"Running protein design with specification {opt.num_aa}")
    start = time.time()
    num_aa, resi, chain, is_cyclic, cyclic_mask = parse_num_aa(opt.num_aa)
    base_chain = chain
    if opt.merge_chains == "True":
        chain = jnp.zeros_like(chain)

    config = getattr(config_choices, opt.config)
    config.eval = True
    config.temperature = opt.temperature
    key = jax.random.PRNGKey(opt.jax_seed)
    model = hk.transform(model_step(config)).apply
    model = jax.jit(model)

    print("Loading model parameters...")
    start = time.time()
    with open(opt.params, "rb") as f:
        params = pickle.load(f)
    print(f"Model parameters loaded in {time.time() - start:.3f} seconds.")

    print("Setting up inputs...")
    init_pos = jnp.zeros((num_aa, 14, 3), dtype=jnp.float32)
    init_pos_mask = jnp.ones((num_aa, 14), dtype=jnp.bool_)
    pos_mask = jnp.zeros((num_aa,), dtype=jnp.bool_)
    init_aa_gt = 20 * jnp.ones((num_aa,), dtype=jnp.int32)
    print(f"Inputs set up in {time.time() - start:.3f} seconds.")
    print(f"Start generating {opt.num_designs} designs with denoising steps...")
    for idx in range(opt.num_designs):
        key, subkey = jax.random.split(key, 2)
        data = dict(
            all_atom_positions=init_pos,
            all_atom_mask=init_pos_mask,
            condition=dict(
                pos=init_pos[:, :5],
                pos_mask=pos_mask,
                dssp=dssp(50, 0.5),
                hotspots=jnp.zeros(chain.shape, dtype=jnp.bool_),
            ),
            aa_gt=init_aa_gt,
            residue_index=resi,
            chain_index=chain,
            batch_index=jnp.zeros_like(chain),
            mask=jnp.ones_like(chain, dtype=jnp.bool_),
            seq_mask=jnp.ones_like(chain, dtype=jnp.bool_),
            residue_mask=jnp.ones_like(chain, dtype=jnp.bool_),
        )
        result = model(params, subkey, data)
        aatype = result["aatype"]
        print(decode_sequence(aatype))
