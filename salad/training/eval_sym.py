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

from salad.aflib.common.protein import to_pdb, Protein
from salad.aflib.model.all_atom_multimer import atom14_to_atom37, get_atom37_mask

from salad.modules.noise_schedule_benchmark import (
    StructureDiffusionPredict, StructureDiffusionNoise, sigma_scale_cosine, sigma_scale_framediff)
from salad.modules.config import noise_schedule_benchmark as config_choices
from salad.modules.utils.geometry import index_mean
from salad.modules.utils.dssp import assign_dssp
from flexloop.utils import parse_options

class Screw:
    def __init__(self, count=3, angle=0, radius=15.0, translation=0.0, chain_center=False):
        angle = angle / 180 * np.pi
        self.count = count
        self.angle = angle
        self.radius = radius
        self.translation = translation
        self.chain_center = chain_center
        # prepare rotation matrices
        self.rot = jnp.array([
            [np.cos(angle),  0.0000000, -np.sin(angle)],
            [    0.0000000,  1.0000000,      0.0000000],
            [np.sin(angle),  0.0000000,  np.cos(angle)]])
        self.rot_x = [jnp.eye(3), self.rot]
        for _ in range(count - 2):
            self.rot_x.append(jnp.einsum("ac,cb->ab", self.rot_x[-1], self.rot))

    def replicate_pos(self, first: jnp.ndarray, do_radius: bool):
        # replicate positions
        first = first.at[:].add(jnp.array([do_radius * self.radius, 0.0, 0.0]))
        replicates = []
        for idx in range(self.count):
            replicates.append(
                jnp.einsum("...c,cd->...d", first, self.rot_x[idx])
            + idx * jnp.array([0.0, self.translation, 0.0]))
        result = jnp.concatenate(replicates, axis=0)
        return result - result[:, 1].mean(axis=0)

    def couple_pos(self, x: jnp.ndarray, do_radius: bool):
        # couple positions
        size = x.shape[0] // self.count
        representative = 0
        for idx in range(self.count):
            representative += jnp.einsum(
                "...c,cd->...d",
                cyclic_centering(x[idx * size:(idx + 1) * size], docenter=do_radius),
                self.rot_x[idx].T)
        representative /= self.count
        return representative

    def select_pos(self, x: jnp.ndarray, do_radius: bool):
        size = x.shape[0] // self.count
        idx = self.count // 2
        representative = jnp.einsum(
            "...c,cd->...d",
            cyclic_centering(x[idx * size:(idx + 1) * size], docenter=do_radius),
            self.rot_x[idx].T)
        return representative#jnp.where(do_radius, representative, x[idx * size:(idx + 1) * size])

    def replicate_features(self, data: jnp.ndarray):
        return jnp.concatenate(self.count * [data], axis=0)

    def couple_features(self, data: jnp.ndarray):
        size = data.shape[0]
        subsize = size // self.count
        units = data.reshape(self.count, subsize, *data.shape[1:])
        unit = units.mean(axis=0)
        return unit

    def select_features(self, data):
        size = data.shape[0]
        subsize = size // self.count
        idx = self.count // 2
        unit = data[idx * subsize:(idx + 1) * subsize]
        return unit

class ScrewVP:
    def __init__(self, count=3, angle=0, radius=15.0, translation=0.0, chain_center=False):
        angle = angle / 180 * np.pi
        self.count = count
        self.angle = angle
        self.radius = radius
        self.translation = translation
        self.chain_center = chain_center
        # prepare rotation matrices
        self.rot = jnp.array([
            [np.cos(angle),  0.0000000, -np.sin(angle)],
            [    0.0000000,  1.0000000,      0.0000000],
            [np.sin(angle),  0.0000000,  np.cos(angle)]])
        self.rot_x = [jnp.eye(3), self.rot]
        for _ in range(count - 2):
            self.rot_x.append(jnp.einsum("ac,cb->ab", self.rot_x[-1], self.rot))

    def mean_center(self, x):
        if self.translation != 0.0:
            center = x[:, 1, 1].mean(axis=0)
            x = x.at[:, :, 1].add(-center)
        return x

    def replicate_pos(self, first: jnp.ndarray):
        # replicate positions
        replicates = []
        for idx in range(self.count):
            replicates.append(
                jnp.einsum("...c,cd->...d", first, self.rot_x[idx]) +
                idx * self.translation * jnp.array([0.0, 1.0, 0.0]))
        result = jnp.concatenate(replicates, axis=0)
        return result - result[:, 1].mean(axis=0)

    def couple_pos(self, x: jnp.ndarray):
        # couple positions
        size = x.shape[0] // self.count
        representative = 0
        for idx in range(self.count):
            representative += self.mean_center(jnp.einsum(
                "...c,cd->...d",
                x[idx * size:(idx + 1) * size],
                self.rot_x[idx].T))
        representative /= self.count
        return representative

    def select_pos(self, x: jnp.ndarray):
        size = x.shape[0] // self.count
        idx = self.count // 2
        representative = jnp.einsum(
            "...c,cd->...d",
            x[idx * size:(idx + 1) * size],
            self.rot_x[idx].T)
        return self.mean_center(representative)

    def replicate_features(self, data: jnp.ndarray):
        return jnp.concatenate(self.count * [data], axis=0)

    def couple_features(self, data: jnp.ndarray):
        size = data.shape[0]
        subsize = size // self.count
        units = data.reshape(self.count, subsize, *data.shape[1:])
        unit = units.mean(axis=0)
        return unit

    def select_features(self, data):
        size = data.shape[0]
        subsize = size // self.count
        idx = self.count // 2
        unit = data[idx * subsize:(idx + 1) * subsize]
        return unit

# TODO refactor: move common step-components to their own module
def model_step(config, sym_op: Screw):
    config = deepcopy(config)
    config.eval = True
    def step(data, prev):
        noise = StructureDiffusionNoise(config)
        predict = StructureDiffusionPredict(config)
        # apply gradients & symmetrise inputs
        pos = data["pos"]
        ts = data["t_pos"][0]
        # decide if chains should be fixed at radius
        do_radius = ts > (config.fixcenter_threshold or 40.0)
        if config.mix_output == "couple":
            mix_output = sym_op.couple_pos
        else:
            mix_output = sym_op.select_pos
        pos = sym_op.replicate_pos(mix_output(pos, do_radius), do_radius)
        # potentially apply compacting update
        ca = pos[:, 1]
        resi = data["residue_index"]
        chain = data["chain_index"]
        num_aa = chain.shape[0]
        num_rep = sym_op.count
        same_chain = chain[:, None] == chain[None, :]
        mean_pos = ca.reshape(num_rep, num_aa // num_rep, 3).mean(axis=1, keepdims=True)
        mean_pos = jnp.broadcast_to(mean_pos, (num_rep, num_aa // num_rep, 3)).reshape(-1, 3)
        #index_mean(ca, chain, data["mask"][:, None])
        compact_update = mean_pos[:, None] - pos
        rdist = jnp.where(same_chain, abs(resi[:, None] - resi[None, :]), jnp.inf)
        long_range = rdist > 16
        # clashyness metric to take gradients over
        def clashy(x):
            clash_threshold = 8.0
            dist = jnp.sqrt(jnp.maximum(
                ((x[:, None] - x[None, :]) ** 2).sum(axis=-1), 1e-6))
            clashyness = jnp.where(
                long_range,
                jax.nn.relu(clash_threshold - dist) / clash_threshold, 0).sum()
            return clashyness
        clash_update = -jax.grad(clashy, argnums=(0,))(ca)[0][:, None]
        compact_lr = config.compact_lr
        clash_lr = config.clash_lr
        # clash_norm = jnp.linalg.norm(clash_update)
        # jax.debug.print("{clash_lr} {compact_lr} {clash_norm}",
        #                 clash_lr=clash_lr, compact_lr=compact_lr, clash_norm=clash_norm)
        pos += compact_lr * data["t_pos"][:, None, None] * compact_update
        clash_step = clash_lr * data["t_pos"][:, None, None] * clash_update
        # clash_norm = jnp.linalg.norm(clash_step)
        # jax.debug.print("{ts} {clash_norm}",
        #                  ts=ts, clash_norm=clash_norm)
        pos += clash_step#clash_lr * data["t_pos"][:, None, None] * clash_update
        data["pos"] = pos
        if "dssp_condition" in data:
            data["dssp_condition"] = sym_op.replicate_features(
                sym_op.select_features(data["dssp_condition"]))
        # apply noise
        data.update(noise(data))
        # symmetrise noise
        pos_noised = data["pos_noised"]
        if config.replicate_noise:
            pos_noised = sym_op.replicate_pos(sym_op.select_pos(pos_noised, do_radius), do_radius)
        data["pos_noised"] = pos_noised
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
    return jnp.where(docenter, x.at[:].add(-x[:, 1].mean(axis=0)), x)

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

def eval_dssp(pos, replicate_count):
    L = pos.shape[0] // replicate_count
    pos = pos[:L]
    dssp, _, _ = assign_dssp(
        pos[:, :4], jnp.zeros((L,), dtype=jnp.int32), mask=jnp.ones((L,), dtype=jnp.bool_))
    loop, helix, strand = jax.nn.one_hot(dssp, 3, axis=-1).mean(axis=0)
    return dict(L=loop, H=helix, E=strand)

def eval_squish(aa):
    seq = decode_sequence(aa)
    return (seq.count("A") + seq.count("G")) / len(seq)

if __name__ == "__main__":
    opt = parse_options(
        "sample cyclic proteins from a protein diffusion model.",
        params="checkpoint.jax",
        out_path="outputs/",
        config="default_ve_scaled",
        timescale_pos="ve(t)",
        timescale_seq="1",
        num_aa="100:100:100:100",
        sym="False",
        merge_chains="False",
        # number of replicas of a repeat subunit
        replicate_count=4,
        # angle of rotation around the symmetry axis
        screw_angle=90.0,
        # center of mass translation along the symmetry axis
        screw_translation=0.0,
        # center of mass distance from the symmetry axis
        screw_radius=0.0,
        # threshold noise level below which subunits are not centered
        fixcenter_threshold=0.0001,
        # learning rate for compactness gradients
        compact_lr=0.0,
        # learning rate for clash gradients
        clash_lr=0.0,
        # filter designs with ALA/GLY fraction > f_small
        f_small=1.0,
        # filter designs with beta-strand fraction > f_strand
        f_strand=1.0,
        sym_noise="True",
        num_designs=10,
        num_steps=500,
        out_steps="499",
        dssp_mean="none",
        dssp="none",
        mode="screw",
        prev_threshold=0.9,
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
    config.replicate_noise = opt.sym_noise == "True"
    config.compact_lr = opt.compact_lr
    config.clash_lr = opt.clash_lr
    config.fixcenter_threshold = opt.fixcenter_threshold

    if opt.mode == "rotation":
        sym_op_type = ScrewVP
    if opt.mode == "screw":
        sym_op_type = Screw
    sym_op = sym_op_type(
        opt.replicate_count, opt.screw_angle,
        opt.screw_radius, opt.screw_translation)
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
    idx = 0
    while idx < opt.num_designs:
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
                alagly = eval_squish(data["aatype"])
                if alagly > opt.f_small:
                    print(f"failed, too much ALA / GLY {alagly * 100:.1f}")
                    break
                edssp = eval_dssp(data["pos"], opt.replicate_count)
                if edssp["E"] > opt.f_strand:
                    print(f"failed, too much strand {100 * edssp['E']:.1f} %")
                    break
                atom37 = atom14_to_atom37(data["atom_pos"], data["aatype"])
                atom37_mask = get_atom37_mask(data["aatype"])
                atom37_mask *= data["mask"][:, None]
                protein = Protein(np.array(atom37), np.array(data["aatype"]),
                                np.array(atom37_mask), np.array(data["residue_index"]),
                                np.array(base_chain), maxprob[:, None] * np.array(
                                    jnp.ones_like(atom37_mask, dtype=jnp.float32)))
                pdb_string = to_pdb(protein)
                sequence = decode_sequence(data["seq"])
                with open(f"{opt.out_path}/result_{idx}_{step}.pdb", "wt") as f:
                    f.write(pdb_string)
                print(f"Design {idx} generated in {time.time() - start:.3f} seconds.")
                print(f"With ALA/GLY percentage: {alagly * 100:.1f} % and secondary structure "
                      f"H: {100 * edssp['H']:.1f} %, E: {100 * edssp['E']:.1f} %, L: {100 * edssp['L']:.1f} %")
                if step == out_steps[-1]:
                    idx += 1
                    break
    print("All designs successfully generated.")
