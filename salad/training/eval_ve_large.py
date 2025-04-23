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

from salad.modules.noise_schedule_benchmark import (
    StructureDiffusionInference, sigma_scale_cosine, sigma_scale_framediff,
    positions_to_ncacocb)
from salad.modules.config import noise_schedule_benchmark as config_choices
from flexloop.utils import parse_options
from salad.modules.utils.geometry import extract_aa_frames
from salad.training.eval_s2s import decode_sequence

def model_step(config):
    module = StructureDiffusionInference
    config = deepcopy(config)
    config.eval = True
    def step(data, prev):
        diffusion = module(config)
        start_prev = prev
        out, prev = diffusion(data, start_prev)
        return out, prev
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

def parse_out_steps(data):
    data = data.strip()
    if data == "all":
        return range(1000)
    return [int(c) for c in data.split(",")]

def sigma_scale_ve(t, sigma_max=80.0, rho=7.0):    
    sigma_min = 0.05
    time = (sigma_max ** (1 / rho) + (1 - t) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return time

def sigma_scale_edm(t):
    pos_mean_sigma = 1.6
    pos_std_sigma = 1.4
    return jnp.exp(pos_mean_sigma + pos_std_sigma * jax.scipy.stats.norm.ppf(jnp.clip(t, 0.001, 0.999)).astype(jnp.float32))

def parse_timescale(data):
    data = data.strip()
    predefined = dict(
        cosine=sigma_scale_cosine,
        framediff=sigma_scale_framediff,
        ve=sigma_scale_ve,
        edm=sigma_scale_edm,
    )
    return lambda t: eval(data, None, dict(t=t, np=np, **predefined))

def cloud_std_default(num_aa):
    minval = num_aa ** 0.4
    return minval + np.random.rand() * 3

def parse_cloud_std(data):
    data = data.strip()
    if data == "none":
        return lambda num_aa: None
    predefined = dict(
        default=cloud_std_default
    )
    return lambda num_aa: eval(data, None, dict(num_aa=num_aa, np=np, **predefined))

def random_dssp_mean(max_loop=0.5):
    helix = np.random.random()
    sheet = 1 - helix
    loop = np.random.random() * max_loop
    return np.array([loop, (1 - loop) * helix, (1 - loop) * sheet])

def random_dssp(count, p=0.5):
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
    helices = ["_" + "H" * (num - 2) + "_" if random.random() > p else "_" * num for num in helices]
    sheets = ["_" + "E" * (num - 2) + "_" if random.random() > p else "_" * num for num in sheets]
    loops = ["L" * num if random.random() < p else "_" * num for num in loops]
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

if __name__ == "__main__":
    opt = parse_options(
        "sample from a protein diffusion model.",
        params="params/default_ve_scaled-200k.jax",
        out_path="outputs/",
        config="default_ve_scaled",
        timescale_pos="ve(t)",
        timescale_seq="1",
        num_aa="100",
        sym="False",
        merge_chains="False",
        sequence_only="False",
        renoise="False",
        sample_final="False",
        replicate_count=3,
        screw_angle=30.0,
        screw_translation=15.0,
        screw_radius=0.0,
        sym_mean="True",
        num_designs=10,
        num_steps=500,
        out_steps="499",
        dssp_mean="none",
        dssp="none",
        template="none",
        template_aa="False",
        prev_threshold=1.0,
        cloud_std="none",
        depth_adjust="none",
        sym_threshold=0.0,
        start_std=10.0,
        start_lr=0.1,
        start_steps=100,
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
    out_steps = parse_out_steps(opt.out_steps)

    config = getattr(config_choices, opt.config)
    config.eval = True
    config.cyclic = is_cyclic
    if opt.depth_adjust != "none":
        config.diffusion_depth = int(opt.depth_adjust)
    key = jax.random.PRNGKey(opt.jax_seed)
    _, model = hk.transform(model_step(config))
    model = jax.jit(model)
    final_config = deepcopy(config)
    final_config.sample_aa = True
    final_config.temperature = 0.1
    model_final = jax.jit(hk.transform(model_step(final_config)).apply)

    print("Loading model parameters...")
    start = time.time()
    with open(opt.params, "rb") as f:
        params = pickle.load(f)
    print(f"Model parameters loaded in {time.time() - start:.3f} seconds.")

    print("Setting up inputs...")
    if dssp is not None and not isinstance(dssp, Callable):
        num_aa = dssp.shape[0]
    init_aa_gt = 20 * jnp.ones((num_aa,), dtype=jnp.int32)
    init_local = jnp.zeros((num_aa, config.local_size), dtype=jnp.float32)
    cloud_std_func = parse_cloud_std(opt.cloud_std)
    print(f"Inputs set up in {time.time() - start:.3f} seconds.")
    print(f"Start generating {opt.num_designs} designs with {opt.num_steps} denoising steps...")
    os.makedirs(opt.out_path, exist_ok=True)
    for idx in range(opt.num_designs):
        # directions = np.random.randn(64, 3)
        # directions /= np.linalg.norm(directions, axis=-1)[..., None]
        # init_centers = np.cumsum(directions * 10.0, axis=0)
        start = opt.start_std * np.random.randn(5, 3)
        start = np.cumsum(start, axis=0)
        def cost(x):
            dist = jnp.sqrt(1e-6 + ((x[:, None] - x[None, :]) ** 2).sum(axis=-1))
            return ((1 - jnp.eye(5)) * (dist - 10) ** 2).mean()
        current = start
        for _ in range(opt.start_steps):
            grad = jax.grad(cost, argnums=0)(current)
            current -= opt.start_lr * grad
        init_pos = np.repeat(current, 200, axis=0)[:num_aa]
        init_pos = np.repeat(init_pos[:, None], 5 + config.augment_size, axis=1)
        init_pos = jnp.array(init_pos)
        # init_pos = jnp.zeros((num_aa, 14 if config.encode_atom14 else 5 + config.augment_size, 3), dtype=jnp.float32)
        cloud_std = cloud_std_func(num_aa)
        data = dict(
            pos=init_pos,
            aa_gt=init_aa_gt,
            seq=init_aa_gt,
            residue_index=resi,
            chain_index=chain,
            batch_index=jnp.zeros_like(chain),
            mask=jnp.ones_like(chain, dtype=jnp.bool_),
            cyclic_mask=cyclic_mask,
            t_pos=jnp.ones((num_aa,), dtype=jnp.float32),
            t_seq=jnp.ones((num_aa,), dtype=jnp.float32)
        )
        if cloud_std is not None:
            data["cloud_std"] = cloud_std
        if dssp_mean is not None:
            if isinstance(dssp_mean, Callable):
                data["dssp_mean"] = dssp_mean()
            else:
                data["dssp_mean"] = dssp_mean
        if dssp is not None:
            if isinstance(dssp, Callable):
                data["dssp_condition"] = dssp(num_aa)
            else:
                data["dssp_condition"] = dssp
        init_prev = dict(
            pos=jnp.zeros_like(init_pos),
            local=init_local
        )
        prev = init_prev
        start = time.time()
        for step in range(opt.num_steps):
            key, subkey = jax.random.split(key)
            # FIXME: step / num_steps or step / (num_steps - 1)
            raw_t = 1 - step / opt.num_steps
            # raw_t = 1 - step / (opt.num_steps - 1)
            scaled_t = parse_timescale(opt.timescale_pos)(raw_t)
            #print(raw_t, scaled_t)
            data["t_pos"] = scaled_t * jnp.ones_like(data["t_pos"])
            data["t_seq"] = parse_timescale(opt.timescale_seq)(raw_t) * jnp.ones_like(data["t_seq"])
            # FIXME: was ist scaled_t before? It likely was not!
            # if scaled_t < opt.prev_threshold:
            #     prev = init_prev
            if raw_t < opt.prev_threshold:
                prev = init_prev
            if opt.sample_final == "True" and step in out_steps:
                update, prev = model_final(params, subkey, data, prev)
            else:
                update, prev = model(params, subkey, data, prev)
            pos = update["pos"]
            data["pos"] = pos
            data["seq"] = jnp.argmax(update["aa"], axis=-1)
            maxprob = jax.nn.softmax(update["aa"], axis=-1).max(axis=-1)
            data["atom_pos"] = update["atom_pos"]
            data["aatype"] = update["aatype"]
            if step in out_steps:
                results = [{key: value for key, value in data.items()}]
                if opt.renoise == "True":
                    for idr in range(9):
                        for s in range(int(opt.num_steps * 0.6), step):
                            key, subkey = jax.random.split(key)
                            raw_t = 1 - s / opt.num_steps
                            scaled_t = parse_timescale(opt.timescale_pos)(raw_t)
                            data["t_pos"] = scaled_t * jnp.ones_like(data["t_pos"])
                            data["t_seq"] = parse_timescale(opt.timescale_seq)(raw_t) * jnp.ones_like(data["t_seq"])
                            prev = init_prev
                            update, prev = model(params, subkey, data, prev)
                            pos = update["pos"]
                            data["pos"] = pos
                            data["seq"] = jnp.argmax(update["aa"], axis=-1)
                            maxprob = jax.nn.softmax(update["aa"], axis=-1).max(axis=-1)
                            data["atom_pos"] = update["atom_pos"]
                            data["aatype"] = update["aatype"]
                        item = {key:value for key, value in data.items()}
                        results.append(item)
                for idr, data in enumerate(results):
                    if opt.sequence_only == "True":
                        header = f">design_{idx}_{step}\n"
                        if opt.renoise == "True":
                            header = f">design_{idx}_{idr}_{step}\n"
                        with open(f"{opt.out_path}/sequences.fa", "at") as f:
                            f.write(header)
                            f.write(decode_sequence(data["aatype"],
                                                    jnp.ones_like(chain, dtype=jnp.bool_)) + "\n")
                            f.flush()
                    else:
                        pdb_path = f"{opt.out_path}/result_{idx}_{step}.pdb"
                        if opt.renoise == "True":
                            pdb_path = f"{opt.out_path}/result_{idx}_{idr}_{step}.pdb"
                        atom37 = atom14_to_atom37(data["atom_pos"], data["aatype"])
                        atom37_mask = get_atom37_mask(data["aatype"])
                        protein = Protein(np.array(atom37), np.array(data["aatype"]),
                                          np.array(atom37_mask), np.array(data["residue_index"]),
                                          np.array(base_chain), maxprob[:, None] * np.array(
                                            np.ones_like(atom37_mask, dtype=jnp.float32)))
                        pdb_string = to_pdb(protein)
                        with open(pdb_path, "wt") as f:
                            f.write(pdb_string)
                if step == out_steps[-1]:
                    break
        print(f"Design {idx} generated in {time.time() - start:.3f} seconds.")
    print("All designs successfully generated.")
