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

from salad.aflib.common.protein import from_pdb_string, to_pdb, Protein
from salad.aflib.model.all_atom_multimer import atom14_to_atom37, get_atom37_mask

from salad.modules.noise_schedule_benchmark import (
    StructureDiffusionPredict, StructureDiffusionNoise, sigma_scale_cosine, sigma_scale_framediff)
from salad.modules.config import noise_schedule_benchmark as config_choices
from salad.modules.utils.geometry import compute_pseudo_cb, index_align
from flexloop.utils import parse_options

def read_structures(c, path):
    parent = read_pdb(c, path + "_s0.pdb")
    child_1 = read_pdb(c, path + "_s1.pdb")
    child_2 = read_pdb(c, path + "_s2.pdb")
    return parent, child_1, child_2

def read_pdb(c, path):
    with open(path, "rt") as f:
        protein = from_pdb_string(f.read())
        atom37 = protein.atom_positions
        ncaco = np.concatenate((atom37[:, :3], atom37[:, 4:5]), axis=1)
        cb = compute_pseudo_cb(ncaco)
        ncacocb = np.concatenate((ncaco, cb[:, None]), axis=1)
    return np.concatenate((
        ncacocb, np.zeros((ncacocb.shape[0], c.augment_size, 3), dtype=ncacocb.dtype)), axis=1)

def couple_part(x, y, mask):
    # y aligned to x
    y_at_x = index_align(
        y, x,
        jnp.zeros((y.shape[0],), dtype=jnp.int32),
        jnp.ones((y.shape[0],), dtype=jnp.int32) * mask)
    # x aligned to y
    x_at_y = index_align(
        x, y,
        jnp.zeros((y.shape[0],), dtype=jnp.int32),
        jnp.ones((y.shape[0],), dtype=jnp.int32) * mask)
    x = jnp.where(mask[:, None, None], y_at_x, x)
    y = jnp.where(mask[:, None, None], (y * 0.5 + x_at_y * 0.5), y)
    return x, y

class SplitChange:
    def __init__(self, same_mask, parent_split):
        self.same_mask = same_mask
        self.parent_split = parent_split

    def update_pos(self, parent, c_1, c_2, time):
        p_1, p_2 = parent[:self.parent_split], parent[self.parent_split:]
        m_1, m_2 = self.same_mask[:self.parent_split], self.same_mask[self.parent_split:]
        c_1, _ = couple_part(c_1, p_1, m_1)
        c_2, _ = couple_part(c_2, p_2, m_2)
        parent = jnp.concatenate((p_1, p_2), axis=0)
        return parent, c_1, c_2

def model_step(config, move_op: SplitChange):
    config = deepcopy(config)
    config.eval = True
    def step(data, prev):
        noise = StructureDiffusionNoise(config)
        predict = StructureDiffusionPredict(config)
        parent = data[0]["pos"]
        child_1 = data[1]["pos"]
        child_2 = data[2]["pos"]
        parent, child_1, child_2 = move_op.update_pos(
            parent, child_1, child_2, None)
        data[0]["pos"] = parent
        data[1]["pos"] = child_1
        data[2]["pos"] = child_2
        # prev[0]["pos"] = parent
        # prev[1]["pos"] = child_1
        # prev[2]["pos"] = child_2
        out = []
        new_prev = []
        for d, p in zip(data, prev):
            d["pos"] = center(d["pos"])
            d.update(noise(d))
            o, p = predict(d, p)
            out.append(o)
            new_prev.append(p)
        return tuple(out), tuple(new_prev)
    return step

def center(x):
    cc = x[:, 1].mean(axis=0)
    return x - cc[None, None, :]

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

def parse_out_steps(data):
    data = data.strip()
    if data == "all":
        return range(1000)
    return [int(c) for c in data.split(",")]

def sigma_scale_ve(t, sigma_max=80.0, rho=7.0):    
    sigma_min = 0.05
    time = (sigma_max ** (1 / rho) + (1 - t) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return time

def parse_timescale(data):
    data = data.strip()
    predefined = dict(
        cosine=sigma_scale_cosine,
        framediff=sigma_scale_framediff,
        ve=sigma_scale_ve
    )
    return lambda t: eval(data, None, dict(t=t, np=np, **predefined))

def parse_dssp(data):
    DSSP_CODE = "LHE_"
    return np.array([DSSP_CODE.index(c) for c in data.strip()], dtype=np.int32)

def parse_parent(data: str):
    parent, child_1, child_2 = data.strip().split("/")
    parent_split = len(child_1)
    same_mask = np.array([c.isupper() for c in parent], dtype=np.bool_)
    parent = parse_dssp(parent.upper())
    child_1 = parse_dssp(child_1.upper())
    child_2 = parse_dssp(child_2.upper())
    move_op = SplitChange(same_mask, parent_split)
    dssp = (parent, child_1, child_2)
    return move_op, dssp

def decode_sequence(x):
    AA_CODE="ARNDCQEGHILKMFPSTWYVX"
    return "".join([AA_CODE[c] for c in x])

def make_data(c, chains=None, dssp=None):
    total = sum(chains)
    init_pos = jnp.zeros((total, 5 + c.augment_size, 3), dtype=jnp.float32)
    init_aa_gt = 20 * jnp.ones((total,), dtype=jnp.int32)
    resi = jnp.arange(total, dtype=jnp.int32)
    chain = jnp.concatenate([
        np.array([i] * c, dtype=jnp.int32)
        for i, c in enumerate(chains)], axis=0)
    batch = jnp.zeros_like(chain)
    mask = jnp.ones_like(batch, dtype=jnp.bool_)
    result = dict(
        pos=init_pos,
        aa_gt=init_aa_gt,
        seq=init_aa_gt,
        residue_index=resi,
        chain_index=chain,
        batch_index=batch,
        mask=mask,
        cyclic_mask=jnp.zeros_like(mask),
        t_pos=jnp.ones((total,), dtype=jnp.float32),
        t_seq=jnp.ones((total,), dtype=jnp.float32)
    )
    if dssp is not None:
        result["dssp_condition"] = dssp
    return result

def make_prev(c, chains=None):
    total = sum(chains)
    init_pos = jnp.zeros((total, 5 + c.augment_size, 3), dtype=jnp.float32)
    init_local = jnp.zeros((total, c.local_size), dtype=jnp.float32)
    return dict(
        pos=jnp.zeros_like(init_pos),
        local=init_local
    )

if __name__ == "__main__":
    opt = parse_options(
        "sample from a protein diffusion model.",
        params="checkpoint.jax",
        in_path="inputs/",
        config="default",
        timescale_pos="cosine(t)",
        timescale_seq="1",
        num_aa="100",
        sym="False",
        merge_chains="False",
        variants=10,
        sym_mean="True",
        num_designs=10,
        num_steps=500,
        out_steps="499",
        dssp="none",
        prev_threshold=1.0,
        cloud_std="none",
        sym_threshold=0.0,
        jax_seed=42
    )
    move_op, dssp = parse_parent(opt.dssp)

    print(f"Running protein design with specification {opt.num_aa}")
    start = time.time()
    out_steps = parse_out_steps(opt.out_steps)

    config = getattr(config_choices, opt.config)
    config.eval = True
    config.cyclic = False
    key = jax.random.PRNGKey(opt.jax_seed)
    _, model = hk.transform(model_step(config, move_op))
    model = jax.jit(model)

    print("Loading model parameters...")
    start = time.time()
    with open(opt.params, "rb") as f:
        params = pickle.load(f)
    print(f"Model parameters loaded in {time.time() - start:.3f} seconds.")

    print("Setting up inputs...")
    names = [name.split("_s0")[0] for name in os.listdir(opt.in_path) if "_s0" in name]
    print(f"Inputs set up in {time.time() - start:.3f} seconds.")
    print(f"Start generating {opt.num_designs} designs with {opt.num_steps} denoising steps...")
    for name in names:
        idx = int(name.split("_")[1])
        triple = read_structures(config, f"{opt.in_path}/{name}")
        data = tuple([
            make_data(config, chains=[len(item)], dssp=item)
            for item in dssp])
        for d, p in zip(data, triple):
            d["pos"] = p
        init_prev = tuple([
            make_prev(config, chains=[len(item)])
            for item in dssp])
        prev = init_prev
        start = time.time()
        start_data = data
        step = out_steps[-1]
        for idv in range(1, opt.variants):
            data = deepcopy(start_data)
            t0 = 0.1 + random.random() * 0.3
            remaining = int(t0 * opt.num_steps)
            for substep in range(step - remaining, step):
                key, subkey = jax.random.split(key)
                raw_t = 1 - substep / opt.num_steps
                scaled_t = parse_timescale(opt.timescale_pos)(raw_t)
                for d in data:
                    d["t_pos"] = scaled_t * jnp.ones_like(d["t_pos"])
                    d["t_seq"] = parse_timescale(opt.timescale_seq)(raw_t) * jnp.ones_like(d["t_seq"])
                prev = init_prev
                update, prev = model(params, subkey, data, prev)
                for d, u in zip(data, update):
                    d["pos"] = u["pos"]
                    # d["seq"] = jnp.argmax(u["aa"], axis=-1)
                    maxprob = jax.nn.softmax(u["aa"], axis=-1).max(axis=-1)
                    d["atom_pos"] = u["atom_pos"]
                    d["aatype"] = u["aatype"]
            for ids, d in enumerate(data):
                atom37 = atom14_to_atom37(d["atom_pos"], d["aatype"])
                atom37_mask = get_atom37_mask(d["aatype"])
                atom37_mask *= d["mask"][:, None]
                protein = Protein(np.array(atom37), np.array(d["aatype"]),
                                np.array(atom37_mask), np.array(d["residue_index"]),
                                np.array(d["chain_index"]), np.array(
                                    jnp.ones_like(atom37_mask, dtype=jnp.float32)))
                pdb_string = to_pdb(protein)
                sequence = decode_sequence(d["aatype"])
                print(sequence)
                with open(f"{opt.in_path}/result_{idx}_{idv}_{step}_s{ids}.pdb", "wt") as f:
                    f.write(pdb_string)
        print(f"Design {idx} generated in {time.time() - start:.3f} seconds.")
    print("All designs successfully generated.")
