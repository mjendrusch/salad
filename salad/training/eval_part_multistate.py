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
from salad.modules.utils.geometry import extract_aa_frames

def couple_part(x, y):
    x = Vec3Array.from_array(x)
    frames_x, _ = extract_aa_frames(x)
    x_local = frames_x[:, None, None].apply_inverse_to_point(x[None, :])
    y = Vec3Array.from_array(y)
    frames_y, _ = extract_aa_frames(y)
    y_local = frames_y[:, None, None].apply_inverse_to_point(y[None, :])
    coupled = (x_local + y_local) / 2
    for _ in range(3):
        x = frames_x[:, None, None].apply_to_point(coupled).mean(axis=0)
        frames_x, _ = extract_aa_frames(x)
        y = frames_y[:, None, None].apply_to_point(coupled).mean(axis=0)
        frames_y, _ = extract_aa_frames(y)
    return x, y

def model_step(config):
    config = deepcopy(config)
    config.eval = True
    def step(data, prev):
        noise = StructureDiffusionNoise(config)
        predict = StructureDiffusionPredict(config)
        shared_0 = data[0]["pos"]
        shared_1 = data[1]["pos"]
        child_2 = data[2]["pos"]
        
        shared_mean = (shared_0[:-70] + shared_1[:-70]) / 2
        shared_1 = jnp.where(
            data[1]["t_pos"][:, None, None] > 5.0,
            shared_1.at[:-70].set(shared_mean), shared_1)
        shared_0 = jnp.where(
            data[1]["t_pos"][:, None, None] > 5.0,
            shared_0.at[:-70].set(shared_mean), shared_0)
        data[0]["pos"] = shared_0
        data[1]["pos"] = shared_1
        data[0].update(noise(data[0]))
        data[1].update(noise(data[1]))
        noised_0 = data[0]["pos_noised"]
        noised_1 = data[1]["pos_noised"]
        shared_0 = noised_0[:-70]
        shared_1 = noised_1[:-70]
        # NOTE: can't take the mean over noise, that doesn't fly
        shared_mean = shared_0
        noised_1 = jnp.where(
            data[1]["t_pos"][:, None, None] > 5.0,
            noised_1.at[:-70].set(shared_mean), noised_1)
        noised_0 = jnp.where(
            data[1]["t_pos"][:, None, None] > 5.0,
            noised_0.at[:-70].set(shared_mean), noised_0)
        data[0]["pos_noised"] = noised_0
        out_0, prev_0 = predict(data[0], prev[0])
        data[1]["pos_noised"] = noised_1
        out_1, prev_1 = predict(data[1], prev[1])
        aa_0 = jax.nn.log_softmax(out_0["aa"], axis=-1)
        aa_1 = jax.nn.log_softmax(out_1["aa"], axis=-1)
        aa_0 = aa_0.at[:-70].set((aa_0[:-70] + aa_1[:-70]) / 2)
        aa_1 = aa_0
        out_0["aa"] = aa_0
        out_1["aa"] = aa_1
        return (out_0, out_1), (prev_0, prev_1)
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
def screw_replicate(x, count=3, angle=0, radius=15.0, translation=0.0, chain_center=False, time=0.0, mean=True):
    rotmat = jnp.array([
        [np.cos(angle),  0.0000000, -np.sin(angle)],
        [    0.0000000,  1.0000000,      0.0000000],
        [np.sin(angle),  0.0000000,  np.cos(angle)]])
    # jax.debug.print("angle {angle} matrix {matrix}", angle=angle, matrix=rotmat)
    rotmat_x = [jnp.eye(3), rotmat]
    for idx in range(count - 2):
        rotmat_x.append(jnp.einsum("ac,cb->ab", rotmat_x[-1], rotmat))
    size = x.shape[0] // count
    if mean:
        first = 0
        for idx in range(count):
            first += jnp.einsum(
                "...c,cd->...d",
                cyclic_centering(x[idx * size:(idx + 1) * size]),
                rotmat_x[idx].T)
        first /= count
    else:
        idx = count // 2
        first = jnp.einsum(
            "...c,cd->...d",
            cyclic_centering(x[idx * size:(idx + 1) * size]),
            rotmat_x[idx].T)
    def cost(x):
        dist = jnp.sqrt((jnp.maximum(x[:, None] - x[None, :], 1e-3) ** 2).sum(axis=-1))
        switch = jnp.maximum(1 - ((dist - 2) / 8) ** 6, 1e-6) / jnp.maximum(1 - ((dist - 2) / 8) ** 12, 1e-6)
        return switch.mean()
    grad = jax.grad(cost, argnums=0)(first[:, 1])
    grad = jnp.where(jnp.isnan(grad), 0, grad)
    if chain_center:
        first += time * 5 * grad[:, None]
    first = first.at[:].add(jnp.array([radius, 0.0, 0.0]))
    replicates = []
    for idx in range(count):
        replicates.append(
            jnp.einsum("...c,cd->...d", first, rotmat_x[idx])
          + idx * jnp.array([0.0, translation, 0.0]))
    result = jnp.concatenate(replicates, axis=0)
    return result - result[:, 1].mean(axis=0)

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
def screw_flip_replicate(x, count=4, angle=0, radius=15.0, translation=0.0, chain_center=False, mean=True):
    assert count % 2 == 0
    rotmat = jnp.array([
        [np.cos(angle),  0.0000000, -np.sin(angle)],
        [    0.0000000,  1.0000000,      0.0000000],
        [np.sin(angle),  0.0000000,  np.cos(angle)]])
    flip_mat = rot_x(np.pi)
    # jax.debug.print("angle {angle} matrix {matrix}", angle=angle, matrix=rotmat)
    rotmat_x = [jnp.eye(3), rotmat]
    for idx in range(count - 2):
        rotmat_x.append(jnp.einsum("ac,cb->ab", rotmat_x[-1], rotmat))
    size = x.shape[0] // count
    idx = count // 2
    first = jnp.einsum("...c,cd->...d", jnp.einsum(
        "...c,cd->...d",
        cyclic_centering(x[idx * size:(idx + 1) * size]),
        rotmat_x[idx].T), flip_mat.T if idx % 2 != 0 else jnp.eye(3))
    # first = first.at[:].add(jnp.array([radius, 0.0, 0.0]))
    replicates = []
    for idx in range(count):
        replicates.append(
            jnp.einsum("...c,cd->...d",
                jnp.einsum("...c,cd->...d",
                    first, flip_mat if idx % 2 != 0 else jnp.eye(3)) + jnp.array([radius, 0.0, 0.0]),
                rotmat_x[idx])
          + (idx % 2) * jnp.array([0.0, translation, 0.0]))
    result = jnp.concatenate(replicates, axis=0)
    return result - result[:, 1].mean(axis=0)

def dihedral_replicate(x, count=4, angle=0, radius=15.0, translation=0.0, chain_center=False, mean=True):
    distance = translation
    cyclic_count = count // 2
    size = x.shape[0] // count
    rotmat_x = [rot_y(idx * 2 * np.pi / cyclic_count) for idx in range(cyclic_count)]
    first = cyclic_centering(x[:size])# + jnp.array([0, 0, distance / 2])
    second = apply_rot(rot_x(np.pi), first)
    first += jnp.array([radius, 0.0, 0.0])
    second += apply_rot(rot_y(2 * np.pi / count), jnp.array([radius, 0.0, 0.0]))
    first_unit = jnp.concatenate((first, second), axis=0)# + jnp.array([radius, 0.0, 0.0])
    replicates = []
    for idx in range(cyclic_count):
        replicates.append(
            jnp.einsum("...c,cd->...d", first_unit, rotmat_x[idx]))
    result = jnp.concatenate(replicates, axis=0)
    return result - result[:, 1].mean(axis=0)

def dihedral_replicate_4(x, count=4, angle=60, radius=15.0, translation=15.0, time=1.0, chain_center=False, mean=True):
    distance = translation
    cyclic_count = count // 2
    size = x.shape[0] // count
    rotmat_x = [rot_y(idx * 2 * np.pi / cyclic_count) for idx in range(cyclic_count)]
    first = cyclic_centering(x[:size])# + jnp.array([0, 0, distance / 2])
    def cost(x):
        dist = jnp.sqrt((jnp.maximum(x[:, None] - x[None, :], 1e-3) ** 2).sum(axis=-1))
        switch = jnp.maximum(1 - ((dist - 2) / 8) ** 6, 1e-6) / jnp.maximum(1 - ((dist - 2) / 8) ** 12, 1e-6)
        return switch.mean()
    grad = jax.grad(cost, argnums=0)(first[:, 1])
    grad = jnp.where(jnp.isnan(grad), 0, grad)
    if chain_center:
        first += time * 5 * grad[:, None]
    first += jnp.array([radius, 0.0, 0.0])
    replicates = []
    for idx in range(cyclic_count):
        replicates.append(
            jnp.einsum("...c,cd->...d", first, rotmat_x[idx]))
    first_unit = jnp.concatenate(replicates, axis=0)
    second_unit = apply_rot(rot_x(np.pi), first_unit)
    second_unit = apply_rot(rot_y(angle), second_unit)
    result = jnp.concatenate((first_unit, second_unit + jnp.array([0.0, translation, 0.0])), axis=0)
    return result - result[:, 1].mean(axis=0)

def dihedral_replicate_2(x, count=4, angle=0, radius=15.0, translation=0.0, chain_center=False, mean=True):
    distance = translation
    cyclic_count = count // 2
    size = x.shape[0] // count
    rotmat_x = [rot_y(idx * 2 * np.pi / cyclic_count) for idx in range(cyclic_count)]
    first = 0
    for idx in range(count):
        subunit = cyclic_centering(x[size * idx:size * (idx + 1)])
        subunit = apply_rot(rotmat_x[idx // 2].T, subunit)
        if idx % 2 != 0:
            subunit = apply_rot(rot_x(np.pi).T, subunit)
        first += subunit
    first /= count
    #first += jnp.array([0, 0, distance / 2])
    second = apply_rot(rot_x(np.pi), first)
    first += jnp.array([radius, 0.0, 0.0])
    second += apply_rot(rot_y(2 * np.pi / count), jnp.array([radius, 0.0, 0.0]))
    first_unit = jnp.concatenate((first, second), axis=0)
    #first_unit += jnp.array([radius, 0.0, 0.0])
    replicates = []
    for idx in range(cyclic_count):
        replicates.append(
            jnp.einsum("...c,cd->...d", first_unit, rotmat_x[idx]))
    result = jnp.concatenate(replicates, axis=0)
    return result - result[:, 1].mean(axis=0)

def dihedral_replicate_3(x, count=4, angle=0, radius=15.0, translation=0.0, chain_center=False, mean=True):
    cyclic_count = count // 2
    size = x.shape[0] // count
    rotmat_x = [rot_y(idx * 2 * np.pi / cyclic_count) for idx in range(cyclic_count)]
    first = x[10:size - 10]
    fc0 = x[:10]
    fc1 = x[size-10:size]
    second = apply_rot(rot_x(np.pi), first)
    sc0 = x[size:size+10]
    sc1 = x[2 * size - 10:2 * size]
    # if chain_center:
    #     first = first.at[:].set(first[:, 1].mean(axis=0))
    #first += jnp.array([0, 0, distance / 2])
    # second = apply_rot(rot_x(np.pi), first)
    first_unit = jnp.concatenate((fc0, first, fc1, sc0, second, sc1), axis=0)
    #first_unit += jnp.array([radius, 0.0, 0.0])
    replicates = []
    for idx in range(cyclic_count):
        fu = jnp.einsum("...c,cd->...d", first, rotmat_x[idx])
        su = jnp.einsum("...c,cd->...d", second, rotmat_x[idx])
        f0 = x[2 * idx * size:2 * idx * size + 10]
        f1 = x[(2 * idx + 1) * size - 10:(2 * idx + 1) * size]
        s0 = x[(2 * idx + 1) * size:(2 * idx + 1) * size + 10]
        s1 = x[2 * (idx + 1) * size - 10:2 * (idx + 1) * size]
        replicates.append(
            jnp.concatenate((f0, fu, f1, s0, su, s1), axis=0))
    result = jnp.concatenate(replicates, axis=0)
    return result - result[:, 1].mean(axis=0)

def decode_sequence(x):
    AA_CODE="ARNDCQEGHILKMFPSTWYVX"
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
        dssp="none",
        prev_threshold=1.0,
        cloud_std="none",
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
    config.cyclic = is_cyclic
    if opt.sym == "True":
        config.sym_noise = lambda pos: screw_replicate(
            pos, count=opt.replicate_count,
            angle=opt.screw_angle / 180 * np.pi,
            radius=opt.screw_radius,
            translation=opt.screw_translation,
            chain_center=False,
            mean=False)
    key = jax.random.PRNGKey(opt.jax_seed)
    _, model = hk.transform(model_step(config))
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
    if opt.sym == "True":
        init_pos = screw_replicate(
            init_pos, count=opt.replicate_count,
            angle=opt.screw_angle / 180 * np.pi,
            radius=opt.screw_radius,
            translation=opt.screw_translation,
            chain_center=False,
            mean=opt.sym_mean == "True")
    init_aa_gt = 20 * jnp.ones((num_aa,), dtype=jnp.int32)
    init_local = jnp.zeros((num_aa, config.local_size), dtype=jnp.float32)
    cloud_std_func = parse_cloud_std(opt.cloud_std)
    print(f"Inputs set up in {time.time() - start:.3f} seconds.")
    print(f"Start generating {opt.num_designs} designs with {opt.num_steps} denoising steps...")
    os.makedirs(opt.out_path, exist_ok=True)
    for idx in range(opt.num_designs):
        cloud_std = cloud_std_func(num_aa)
        data = [
            dict(
                pos=init_pos,
                aa_gt=init_aa_gt,
                seq=init_aa_gt,
                residue_index=resi,
                chain_index=chain,
                batch_index=jnp.zeros_like(chain),
                mask=(jnp.ones_like(chain, dtype=jnp.bool_)
                      if i == 0 else
                      jnp.ones_like(chain, dtype=jnp.bool).at[-20:].set(False)),
                cyclic_mask=cyclic_mask,
                t_pos=jnp.ones((num_aa,), dtype=jnp.float32),
                t_seq=jnp.ones((num_aa,), dtype=jnp.float32)
            )
            for i in range(2)
        ]
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
        init_prev = tuple([dict(
            pos=jnp.zeros_like(init_pos),
            local=init_local
        ) for i in range(2)])
        prev = init_prev
        start = time.time()
        for step in range(opt.num_steps):
            key, subkey = jax.random.split(key)
            raw_t = 1 - step / opt.num_steps
            scaled_t = parse_timescale(opt.timescale_pos)(raw_t)
            #print(raw_t, scaled_t)
            for d in data:
                d["t_pos"] = scaled_t * jnp.ones_like(d["t_pos"])
                d["t_seq"] = parse_timescale(opt.timescale_seq)(raw_t) * jnp.ones_like(d["t_seq"])
            if raw_t < opt.prev_threshold:
                prev = init_prev
            update, prev = model(params, subkey, data, prev)
            for d, u in zip(data, update):
                d["pos"] = u["pos"]
                d["seq"] = jnp.argmax(u["aa"], axis=-1)
                maxprob = jax.nn.softmax(u["aa"], axis=-1).max(axis=-1)
                d["atom_pos"] = u["atom_pos"]
                d["aatype"] = u["aatype"]
            if step in out_steps:
                for ids, d in enumerate(data):
                    atom37 = atom14_to_atom37(d["atom_pos"], d["aatype"])
                    atom37_mask = get_atom37_mask(d["aatype"])
                    atom37_mask *= d["mask"][:, None]
                    protein = Protein(np.array(atom37), np.array(d["aatype"]),
                                    np.array(atom37_mask), np.array(d["residue_index"]),
                                    np.array(base_chain), maxprob[:, None] * np.array(
                                        jnp.ones_like(atom37_mask, dtype=jnp.float32)))
                    pdb_string = to_pdb(protein)
                    sequence = decode_sequence(d["seq"])
                    print(sequence)
                    with open(f"{opt.out_path}/result_{idx}_{step}_s{ids}.pdb", "wt") as f:
                        f.write(pdb_string)
                if step == out_steps[-1]:
                    break
        print(f"Design {idx} generated in {time.time() - start:.3f} seconds.")
    print("All designs successfully generated.")
