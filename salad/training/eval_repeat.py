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

from alphafold.model.geometry import Vec3Array, Rigid3Array
from alphafold.common.protein import to_pdb, Protein
from alphafold.model.all_atom_multimer import atom14_to_atom37, get_atom37_mask

from salad.modules.noise_schedule_benchmark import (
    StructureDiffusionPredict, StructureDiffusionNoise, sigma_scale_cosine, sigma_scale_framediff)
from salad.modules.config import noise_schedule_benchmark as config_choices
from flexloop.utils import parse_options
from salad.modules.utils.geometry import apply_alignment, extract_aa_frames, index_align, index_kabsch

def couple_all(*parts):
    coupled = [0 for p in parts]
    for idx, x in enumerate(parts):
        for y in parts:
            xp = index_align(
                y, x,
                jnp.zeros((y.shape[0],), dtype=jnp.int32),
                jnp.ones((y.shape[0],), dtype=jnp.int32))
            coupled[idx] += xp / len(parts)
    return coupled

def replicate_all_sym(*parts):
    y = parts[0]
    x = parts[1]
    kabsch_data = index_kabsch(
        y[:, 1], x[:, 1],
        jnp.zeros((y.shape[0],), dtype=jnp.int32),
        jnp.ones((y.shape[0],), dtype=jnp.int32))
    replicated = [y]
    for i in range(len(parts) - 1):
        rp = apply_alignment(
            replicated[-1], kabsch_data)
        replicated.append(rp)
    return replicated

def replicate_all(*parts):
    representative = parts[0]
    replicated = []
    for x in parts:
        rp = index_align(
            representative, x,
            jnp.zeros((representative.shape[0],), dtype=jnp.int32),
            jnp.ones((representative.shape[0],), dtype=jnp.int32))
        replicated.append(rp)
    return replicated

class Repeatness:
    def __init__(self, count=2, autosym=False, threshold=0.5, scale=1.0):
        self.threshold = threshold
        self.scale = scale
        self.count = count
        self.autosym = autosym

    def update_pos(self, pos, time, couple=True):
        size = pos.shape[0]
        parts = pos.reshape(self.count, -1, *pos.shape[1:])
        if couple:
            parts = couple_all(*parts)
        replicate = replicate_all
        if self.autosym:
            replicate = replicate_all_sym
        parts = replicate(*parts)
        pos = jnp.concatenate(parts, axis=0)
        def cost(x):
            dist = jnp.sqrt((jnp.maximum((x[:, None] - x[None, :]) ** 2, 1e-6)).sum(axis=-1))
            switch = jnp.maximum(1 - ((dist - 2) / 8) ** 6, 1e-6) / jnp.maximum(1 - ((dist - 2) / 8) ** 12, 1e-6)
            return switch.mean()
        grad = jax.grad(cost, argnums=0)(pos[:, 1])
        grad = jnp.where(jnp.isnan(grad), 0, grad)
        # jax.debug.print("grad {grad}", grad=grad)
        update = 5 * time[:, None, None] * grad[:, None, :]
        pos += update#jnp.clip(update, -2, 2)
        return pos - pos[:, 1].mean(axis=0)

def model_step(config, move_op: Repeatness):
    config = deepcopy(config)
    config.eval = True
    def step(data, prev):
        noise = StructureDiffusionNoise(config)
        predict = StructureDiffusionPredict(config)
        # apply gradients
        pos = data["pos"]
        pos = move_op.update_pos(pos, data["t_pos"])
        data["pos"] = pos
        # apply noise
        data.update(noise(data))
        out, prev = predict(data, prev)
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

def parse_timescale(data):
    data = data.strip()
    predefined = dict(
        cosine=sigma_scale_cosine,
        framediff=sigma_scale_framediff,
        ve=sigma_scale_ve
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

def random_dssp(count, p=0.1):
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
    loops = ["L" * num if random.random() > p else "_" * num for num in loops]
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

def cyclic_centering(x, docenter=True):
    if not docenter:
        return x
    return x.at[:].add(-x[:, 1].mean(axis=0))

def rot_x(angle):
    return jnp.array([
        [1.0000000,      0.0000000,      0.0000000],
        [0.0000000,  np.cos(angle), -np.sin(angle)],
        [0.0000000,  np.sin(angle),  np.cos(angle)]])
def rot_y(angle):
    return jnp.array([
        [np.cos(angle),  0.0000000, -np.sin(angle)],
        [    0.0000000,  1.0000000,      0.0000000],
        [np.sin(angle),  0.0000000,  np.cos(angle)]])
def rot_z(angle):
    return jnp.array([
        [np.cos(angle), -np.sin(angle), 0.0000000],
        [np.sin(angle),  np.cos(angle), 0.0000000],
        [0.0000000,          0.0000000, 1.0000000]])
def apply_rot(rot, x):
    return jnp.einsum("...c,cd->...d", x, rot)

def decode_sequence(x):
    AA_CODE="ARNDCQEGHILKMFPSTWYVX"
    return "".join([AA_CODE[c] for c in x])

if __name__ == "__main__":
    opt = parse_options(
        "sample cyclic proteins from a protein diffusion model.",
        params="checkpoint.jax",
        out_path="outputs/",
        config="default",
        timescale_pos="cosine(t)",
        timescale_seq="1",
        autosym="False",
        num_aa="100",
        sym="False",
        merge_chains="False",
        replicate_count=3,
        screw_angle=30.0,
        screw_translation=0.0,
        screw_radius=0.0,
        sym_mean="True",
        num_designs=10,
        num_steps=500,
        out_steps="499",
        dssp_mean="none",
        dssp="none",
        mode="screw",
        prev_threshold=1.0,
        cloud_std="none",
        mix_output="couple",
        sym_threshold=0.0,
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
    config.mix_output = opt.mix_output
    config.cyclic = is_cyclic
    sym_op = Repeatness(opt.replicate_count,
                        autosym=opt.autosym == "True")
    key = jax.random.PRNGKey(opt.jax_seed)
    _, model = hk.transform(model_step(config, sym_op))
    model = jax.jit(model)

    print("Loading model parameters...")
    start = time.time()
    with open(opt.params, "rb") as f:
        params = pickle.load(f)
    print(f"Model parameters loaded in {time.time() - start:.3f} seconds.")

    print("Setting up inputs...")
    if dssp is not None and not isinstance(dssp, Callable):
        num_aa = dssp.shape[0]
    init_pos = jnp.zeros((num_aa, 14 if config.encode_atom14 else 5 + config.augment_size, 3), dtype=jnp.float32)
    init_aa_gt = 20 * jnp.ones((num_aa,), dtype=jnp.int32)
    init_local = jnp.zeros((num_aa, config.local_size), dtype=jnp.float32)
    cloud_std_func = parse_cloud_std(opt.cloud_std)
    print(f"Inputs set up in {time.time() - start:.3f} seconds.")
    print(f"Start generating {opt.num_designs} designs with {opt.num_steps} denoising steps...")
    os.makedirs(opt.out_path, exist_ok=True)
    for idx in range(opt.num_designs):
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
            raw_t = 1 - step / opt.num_steps
            scaled_t = parse_timescale(opt.timescale_pos)(raw_t)
            data["t_pos"] = scaled_t * jnp.ones_like(data["t_pos"])
            if raw_t < opt.prev_threshold:
                prev = init_prev
            update, prev = model(params, subkey, data, prev)
            data["pos"] = update["pos"]
            data["seq"] = jnp.argmax(update["aa"], axis=-1)
            maxprob = jax.nn.softmax(update["aa"], axis=-1).max(axis=-1)
            data["atom_pos"] = update["atom_pos"]
            data["aatype"] = update["aatype"]
            if step in out_steps:
                atom37 = atom14_to_atom37(data["atom_pos"], data["aatype"])
                atom37_mask = get_atom37_mask(data["aatype"])
                atom37_mask *= data["mask"][:, None]
                protein = Protein(np.array(atom37), np.array(data["aatype"]),
                                np.array(atom37_mask), np.array(data["residue_index"]),
                                np.array(base_chain), maxprob[:, None] * np.array(
                                    jnp.ones_like(atom37_mask, dtype=jnp.float32)))
                pdb_string = to_pdb(protein)
                sequence = decode_sequence(data["seq"])
                print(sequence)
                with open(f"{opt.out_path}/result_{idx}_{step}.pdb", "wt") as f:
                    f.write(pdb_string)
                if step == out_steps[-1]:
                    break
        print(f"Design {idx} generated in {time.time() - start:.3f} seconds.")
    print("All designs successfully generated.")
