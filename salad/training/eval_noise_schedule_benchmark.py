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

from alphafold.common.protein import to_pdb, Protein, from_pdb_string
from alphafold.model.geometry import Vec3Array
from alphafold.model.all_atom_multimer import atom14_to_atom37, atom37_to_atom14, get_atom37_mask

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

# def random_dssp(count, p=0.1):
#     loop, helix, sheet = random_dssp_mean()
#     helix = int(helix * count)
#     if helix < 6:
#         helix = 0
#     min_helix_count = 1
#     max_helix_count = helix // 6
#     min_helix_count = min(min_helix_count, max_helix_count)
#     if helix == 0:
#         num_helices = 0
#     elif min_helix_count == max_helix_count:
#         num_helices = min_helix_count
#     else:
#         num_helices = np.random.randint(min_helix_count, max_helix_count)
#     sheet = int(sheet * count)
#     if sheet < 8:
#         sheet = 0
#     loop = count - (helix + sheet)
#     min_sheet_count = 2
#     max_sheet_count = sheet // 4
#     if sheet == 0:
#         num_sheets = 0
#     elif min_sheet_count == max_sheet_count:
#         num_sheets = min_sheet_count
#     else:
#         num_sheets = np.random.randint(min_sheet_count, max_sheet_count)
#     helices = [6 for _ in range(num_helices)] 
#     sheets = [4 for _ in range(num_sheets)]
#     loops = [0 for _ in range(num_helices + num_sheets + 1)]
#     while sum(sheets) < sheet:
#         index = np.random.randint(num_sheets)
#         sheets[index] += 1
#     while sum(helices) < helix:
#         index = np.random.randint(num_helices)
#         helices[index] += 1
#     while sum(loops) < loop:
#         index = np.random.randint(0, len(loops))
#         loops[index] += 1
#     helices = ["_" + "H" * (num - 2) + "_" if random.random() > p else "_" * num for num in helices]
#     sheets = ["_" + "E" * (num - 2) + "_" if random.random() > p else "_" * num for num in sheets]
#     loops = ["L" * num if random.random() > p else "_" * num for num in loops]
#     structured = helices + sheets
#     random.shuffle(structured)
#     dssp = loops[0] + "".join([s + l for s, l in zip(structured, loops[1:])])
#     print(dssp)
#     dssp = parse_dssp(dssp)
#     return dssp

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

def parse_template(data):
    if data == "none":
        return None, None, None, None, None
    path, constring = data.strip().split(",")
    conmask = []
    chain_index = []
    chain = 0
    for c in constring:
        if c == "F":
            conmask.append(True)
            chain_index.append(chain)
        elif c == ":":
            chain += 1
        else:
            conmask.append(False)
            chain_index.append(chain)
    conmask = np.array(conmask, dtype=np.bool_)
    chain_index = np.array(chain_index, dtype=np.int32)
    # conmask = np.array([c == "F" for c in constring], dtype=np.bool_)
    size = len(conmask)
    dmap_mask = conmask[:, None] * conmask[None, :]
    with open(path) as f:
        structure = from_pdb_string(f.read())
    atom14 = atom37_to_atom14(
        structure.aatype,
        Vec3Array.from_array(structure.atom_positions),
        structure.atom_mask)[0].to_array()
    aatype_provided = structure.aatype
    ncacocb = positions_to_ncacocb(atom14)
    ncacocb_provided = ncacocb
    cb_provided = ncacocb[:, 4]
    cb = np.zeros((size, 3), dtype=np.float32)
    cb[conmask] = cb_provided
    ncacocb = np.zeros((size, 5, 3), dtype=np.float32)
    ncacocb[conmask] = ncacocb_provided
    aatype = 20 * np.ones((size,), dtype=np.int32)
    aatype[conmask] = aatype_provided
    frames, _ = extract_aa_frames(Vec3Array.from_array(ncacocb))
    rot = frames.rotation
    inv_rot = rot.inverse()
    omap = (inv_rot[:, None] @ rot[None, :]).to_array()
    omap = omap.reshape(omap.shape[0], omap.shape[0], -1)
    dmap = np.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
    return aatype, dmap, omap, dmap_mask, chain_index

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
        out_steps="400",
        dssp_mean="none",
        dssp="none",
        template="none",
        template_aa="False",
        prev_threshold=0.8,
        cloud_std="none",
        depth_adjust="none",
        sym_threshold=0.0,
        jax_seed=42
    )
    dssp_mean = parse_dssp_mean(opt.dssp_mean)
    dssp = parse_dssp(opt.dssp)
    aa_condition, dmap, omap, dmap_mask, chain_index = parse_template(opt.template)

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
    if dmap is not None:
        num_aa = dmap.shape[0]
        resi = np.arange(num_aa, dtype=np.int32)
        base_chain = chain_index#np.zeros((num_aa,), dtype=np.int32)
        chain = chain_index#np.zeros((num_aa,), dtype=np.int32)
        is_cyclic = False
        cyclic_mask = np.zeros((num_aa,), dtype=np.bool_)
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
        if dmap is not None:
            data["dmap"] = dmap
            data["omap"] = omap
            data["dmap_mask"] = dmap_mask
            if opt.template_aa == "True":
                data["aa_condition"] = aa_condition
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
            if opt.sym == "True" and raw_t > opt.sym_threshold:
                pos = screw_replicate(
                    pos, count=opt.replicate_count,
                    angle=opt.screw_angle / 180 * np.pi,
                    radius=opt.screw_radius,
                    translation=opt.screw_translation,
                    chain_center=raw_t > 0.5,
                    time=raw_t,
                    mean=opt.sym_mean == "True")
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
