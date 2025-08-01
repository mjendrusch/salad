import time
import os
from typing import Callable
import random

import json

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
from salad.modules.utils.dssp import assign_dssp
from salad.modules.utils.geometry import extract_aa_frames
from salad.data.allpdb import decode_sequence

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

def parse_constring(x, num_aa):
    result = []
    for idx in range(num_aa):
        if idx < len(x):
            result.append(x[idx] == "F")
        else:
            result.append(False)
    return np.array(result, dtype=np.bool_)

def sample_constring(motif_groups, segments):
    consubsets = {m: "" for m in motif_groups}
    constrings = {m: "" for m in motif_groups}
    total_aa = 0
    for segment, motif_group in segments:
        segval = segment()
        for m in motif_groups:
            if m == motif_group:
                total_aa += len(segval)
                constrings[m] += segval
                consubsets[m] += segval
                for mm in motif_groups:
                    if mm != m:
                        consubsets[mm] += "X" * len(segval)
            else:
                constrings[m] += "X" * len(segval)
    return total_aa, constrings, consubsets

def validate_constrings(constrings, min_length, max_length):
    result = True
    for _, c in constrings.items():
        result = result and len(c) >= min_length
        result = result and len(c) <= max_length
    return result

def parse_genie_remarks(x):
    def segment_sampler(min_length, max_length):
        def _inner():
            return random.randint(min_length, max_length) * "X"
        return _inner
    def fixed_sampler(start, end):
        return lambda: (end - start + 1) * "F"
    def make_constring_sampler(segments, motif_groups, min_length, max_length):
        def _inner():
            done = False
            while not done:
                total_aa, constrings, consubsets = sample_constring(motif_groups, segments)
                done = validate_constrings(constrings, min_length, max_length)
            print("sampling with constraint strings:")
            for m in motif_groups:
                print(m, constrings[m])
            print(consubsets)
            results = [
                (parse_constring(constrings[m], max_length),
                 parse_constring(consubsets[m], total_aa))
                for m in motif_groups
            ]
            return len(constrings[motif_groups[0]]), results, constrings
        return _inner
    max_length = 0
    segments = []
    segment_range_chain = []
    motif_groups = []
    for line in x.split("\n"):
        if not line.startswith("REMARK 999"):
            continue
        if line.startswith("REMARK 999 INPUT"):
            chain_name = line[18]
            start = int(line[19:23].strip())
            end = int(line[23:27].strip())
            if len(line) < 29:
                motif_group = "A"
            else:
                motif_group = line[28]
            if motif_group == " ":
                motif_group = "A"
            motif_groups.append(motif_group)
            if chain_name == " ":
                segments.append((segment_sampler(start, end), None))
            else:
                segments.append((fixed_sampler(start, end), motif_group))
                segment_range_chain.append((np.arange(start, end + 1, dtype=np.int32), chain_name))
        if line.startswith("REMARK 999 MAXIMUM TOTAL LENGTH"):
            max_length = int(line[37:40])
        if line.startswith("REMARK 999 MINIMUM TOTAL LENGTH"):
            min_length = int(line[37:40])
    motif_groups = list(set(motif_groups))
    return max_length, segment_range_chain, make_constring_sampler(segments, motif_groups, min_length, max_length)

def random_dssp_mean(max_loop=0.5):
    helix = np.random.random()
    sheet = 1 - helix
    loop = np.random.random() * max_loop
    return np.array([loop, (1 - loop) * helix, (1 - loop) * sheet])

def parse_dssp(data):
    DSSP_CODE = "LHE_"
    if data == "none":
        return None
    if data == "random":
        return random_dssp
    return np.array([DSSP_CODE.index(c) for c in data.strip()], dtype=np.int32)

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
    loops = ["_" * num for num in loops]
    structured = helices + sheets
    random.shuffle(structured)
    dssp = loops[0] + "".join([s + l for s, l in zip(structured, loops[1:])])
    dssp = parse_dssp(dssp)
    return dssp

def segment_rearrange(x, resi, chain, segment_range_chain):
    chain_order = np.unique([c[-1] for c in segment_range_chain])
    result = []
    for rng, cid in segment_range_chain:
        cid = np.argmax(chain_order == cid)
        mask = (cid == chain) * (resi[:, None] == rng[None, :]).any(axis=1)
        result.append(x[mask])
    return np.concatenate(result, axis=0)

def parse_multi_motif_spec(motif_spec):
    """
    Parse complex motif specifications like:
    "A:6-82,B:10-45" (multiple chains)
    "A:6-82+B:10-45" (linked motifs) 
    "A:6-82,A:90-120" (multiple motifs same chain)
    """
    motifs = []
    if ',' in motif_spec:
        # Multiple separate motifs
        for spec in motif_spec.split(','):
            chain_id, residue_range = spec.strip().split(':')
            start, end = map(int, residue_range.split('-'))
            motifs.append({
                'chain': chain_id,
                'start': start, 
                'end': end,
                'group': chain_id  # Same chain = same group
            })
    else:
        # Single motif
        chain_id, residue_range = motif_spec.split(':')
        start, end = map(int, residue_range.split('-'))
        motifs.append({
            'chain': chain_id,
            'start': start,
            'end': end, 
            'group': chain_id
        })
    
    return motifs

def parse_direct_multi_motif(pdb_path, motif_spec, scaffold_length, max_length=500):
    """Handle multiple motifs directly from PDB"""
    with open(pdb_path, 'r') as f:
        pdb_string = f.read()
    structure = from_pdb_string(pdb_string)
    
    motifs = parse_multi_motif_spec(motif_spec)
    
    all_motif_indices = []
    all_motif_aatype = []
    all_motif_coords = []
    
    for motif in motifs:
        chain_idx = ord(motif['chain']) - ord('A')
        chain_mask = structure.chain_index == chain_idx
        resi_mask = (structure.residue_index >= motif['start']) & (structure.residue_index <= motif['end'])
        motif_mask = chain_mask & resi_mask
        
        motif_indices = np.where(motif_mask)[0]
        all_motif_indices.extend(motif_indices)
        all_motif_aatype.extend(structure.aatype[motif_indices])
        
    all_motif_indices = np.array(all_motif_indices)
    all_motif_aatype = np.array(all_motif_aatype)
    
    atom14 = atom37_to_atom14(
        structure.aatype,
        Vec3Array.from_array(structure.atom_positions),
        structure.atom_mask)[0].to_array()
    
    motif_atom14 = atom14[all_motif_indices]
    ncacocb = positions_to_ncacocb(motif_atom14)
    dssp_provided, _, _ = assign_dssp(
        ncacocb, np.zeros_like(all_motif_aatype), np.ones_like(all_motif_aatype, dtype=np.bool_))
    dssp_provided = np.array(dssp_provided, dtype=np.int32)
    cb_provided = ncacocb[:, 4]
    
    motif_length = len(all_motif_aatype)
    total_length = motif_length + scaffold_length
    
    def template_sampler(diversify=False):
        if diversify:
            dssp = random_dssp(total_length)
        else:
            dssp = 3 * np.ones((total_length,), dtype=np.int32)
        
        omap = np.zeros((total_length, total_length, 9), dtype=np.float32)
        dmap = np.zeros((total_length, total_length), dtype=np.float32)
        dmap_mask = np.zeros((total_length, total_length), dtype=np.bool_)
        aatype = 20 * np.ones((total_length,), dtype=np.int32)
        
        scaffold_start = scaffold_length // 2
        motif_indices_new = np.arange(scaffold_start, scaffold_start + motif_length)
        
        cb = np.zeros((total_length, 3), dtype=np.float32)
        cb[motif_indices_new] = cb_provided
        
        ncacocb_full = np.zeros((total_length, 5, 3), dtype=np.float32)
        ncacocb_full[motif_indices_new] = ncacocb
        aatype[motif_indices_new] = all_motif_aatype
        dssp[motif_indices_new] = dssp_provided
        
        frames, _ = extract_aa_frames(Vec3Array.from_array(ncacocb_full))
        rot = frames.rotation
        inv_rot = rot.inverse()
        omap_update = (inv_rot[:, None] @ rot[None, :]).to_array()
        omap_update = omap_update.reshape(total_length, total_length, -1)
        
        dmap_update = np.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
        dmap_mask_update = np.zeros((total_length, total_length), dtype=np.bool_)
        dmap_mask_update[np.ix_(motif_indices_new, motif_indices_new)] = True
        
        dmap = jnp.where(dmap_mask_update, dmap_update, dmap)
        omap = jnp.where(dmap_mask_update[..., None], omap_update, omap)
        dmap_mask = dmap_mask_update
        
        mask = np.ones((total_length,), dtype=np.bool_)
        
        # Create constraint string
        motif_str = "F" * motif_length
        scaffold_str = "X" * scaffold_length
        scaffold_pre = scaffold_str[:scaffold_start]
        scaffold_post = scaffold_str[scaffold_start:]
        constrings = {"A": scaffold_pre + motif_str + scaffold_post}
        
        return aatype, dmap, omap, dmap_mask, dssp, mask, constrings
    
    return total_length, template_sampler

def parse_template(path):
    with open(path) as f:
        pdb_string = f.read()
        structure = from_pdb_string(pdb_string)
    resi = structure.residue_index
    chain = structure.chain_index
    atom14 = atom37_to_atom14(
        structure.aatype,
        Vec3Array.from_array(structure.atom_positions),
        structure.atom_mask)[0].to_array()
    aatype_provided = structure.aatype
    ncacocb = positions_to_ncacocb(atom14)
    ncacocb_provided = ncacocb
    dssp_provided, _, _ = assign_dssp(
        ncacocb, np.zeros_like(aatype_provided), np.ones_like(aatype_provided, dtype=np.bool_))
    dssp_provided = np.array(dssp_provided, dtype=np.int32)
    cb_provided = ncacocb[:, 4]
    num_aa, segment_range_chain, conmask_sampler = parse_genie_remarks(pdb_string)
    aatype_provided = segment_rearrange(aatype_provided, resi, chain, segment_range_chain)
    ncacocb_provided = segment_rearrange(ncacocb_provided, resi, chain, segment_range_chain)
    dssp_provided = segment_rearrange(dssp_provided, resi, chain, segment_range_chain)
    def template_sampler(diversify=False):
        true_length, conmasks, constrings = conmask_sampler()
        if diversify:
            dssp = random_dssp(num_aa)
        else:
            dssp = 3 * np.ones((num_aa,), dtype=np.int32)
        omap = np.zeros((num_aa, num_aa, 9), dtype=np.float32)
        dmap = np.zeros((num_aa, num_aa,), dtype=np.float32)
        dmap_mask = np.zeros((num_aa, num_aa,), dtype=np.bool_)
        index = np.arange(num_aa, dtype=np.int32)
        aatype = 20 * np.ones((num_aa,), dtype=np.int32)
        for conmask, consubset in conmasks:
            cb = np.zeros((num_aa, 3), dtype=np.float32)
            cb[conmask] = cb_provided[consubset]
            within_dist = (abs(index[:, None] - index[None, conmask]) < 5).any(axis=1)
            dssp[within_dist] = 3
            dssp[conmask] = dssp_provided[consubset]
            ncacocb = np.zeros((num_aa, 5, 3), dtype=np.float32)
            ncacocb[conmask] = ncacocb_provided[consubset]
            aatype[conmask] = aatype_provided[consubset]
            frames, _ = extract_aa_frames(Vec3Array.from_array(ncacocb))
            rot = frames.rotation
            inv_rot = rot.inverse()
            omap_update = (inv_rot[:, None] @ rot[None, :]).to_array()
            omap_update = omap.reshape(omap.shape[0], omap.shape[0], -1)
            dmap_update = np.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
            dmap_mask_update = conmask[:, None] * conmask[None, :]
            dmap_mask += dmap_mask_update
            dmap = jnp.where(dmap_mask_update, dmap_update, dmap)
            omap = jnp.where(dmap_mask_update[..., None], omap_update, omap)
        dmap_mask = dmap_mask > 0
        mask = np.zeros((num_aa,), dtype=np.bool_)
        mask[:true_length] = True
        return aatype, dmap, omap, dmap_mask, dssp, mask, constrings
    return num_aa, template_sampler

if __name__ == "__main__":
    opt = parse_options(
        "sample from a protein diffusion model.",
        params="checkpoint.jax",
        out_path="outputs/",
        config="default",
        diversify="False",
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
        out_steps="499",
        dssp_mean="none",
        dssp="none",
        template="none",
        template_aa="False",
        prev_threshold=1.0,
        cloud_std="none",
        sym_threshold=0.0,
        jax_seed=42,
        motif_spec="none",
        scaffold_length=50
    )
    
    if opt.motif_spec != "none":
        num_aa, template_sampler = parse_direct_multi_motif(opt.template, opt.motif_spec, opt.scaffold_length)
    else:
        num_aa, template_sampler = parse_template(opt.template)
    
    resi = np.arange(num_aa, dtype=np.int32)
    chain = np.zeros((num_aa,), dtype=np.int32)
    print(f"Running protein design with specification {opt.num_aa}")
    start = time.time()
    out_steps = parse_out_steps(opt.out_steps)

    config = getattr(config_choices, opt.config)
    config.eval = True
    config.cyclic = False
    key = jax.random.PRNGKey(opt.jax_seed)
    _, model = hk.transform(model_step(config))
    model = jax.jit(model)

    print("Loading model parameters...")
    start = time.time()
    with open(opt.params, "rb") as f:
        params = pickle.load(f)
    print(f"Model parameters loaded in {time.time() - start:.3f} seconds.")

    print("Setting up inputs...")
    init_pos = jnp.zeros((num_aa, 14 if config.encode_atom14 else 5 + config.augment_size, 3), dtype=jnp.float32)
    init_aa_gt = 20 * jnp.ones((num_aa,), dtype=jnp.int32)
    init_local = jnp.zeros((num_aa, config.local_size), dtype=jnp.float32)
    cloud_std_func = parse_cloud_std(opt.cloud_std)
    print(f"Inputs set up in {time.time() - start:.3f} seconds.")
    print(f"Start generating {opt.num_designs} designs with {opt.num_steps} denoising steps...")
    os.makedirs(opt.out_path, exist_ok=True)
    with open(f"{opt.out_path}/constrings.jsonl", "wt") as cons_f:
        for idx in range(opt.num_designs):
            aatype, dmap, omap, dmap_mask, dssp, mask, constrings = template_sampler(
                diversify=opt.diversify == "True")
            json.dump(constrings, cons_f)
            cons_f.write("\n")
            cloud_std = cloud_std_func(mask.astype(jnp.int32).sum())
            data = dict(
                pos=init_pos,
                aa_gt=aatype,
                seq=aatype,
                residue_index=resi,
                chain_index=chain,
                batch_index=jnp.zeros_like(chain),
                mask=mask,
                cyclic_mask=jnp.zeros_like(chain, jnp.bool_),
                t_pos=jnp.ones((num_aa,), dtype=jnp.float32),
                t_seq=jnp.ones((num_aa,), dtype=jnp.float32)
            )
            data["dmap"] = dmap
            data["omap"] = omap
            data["dmap_mask"] = dmap_mask
            data["aa_condition"] = aatype
            if cloud_std is not None:
                data["cloud_std"] = cloud_std
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
                data["t_seq"] = parse_timescale(opt.timescale_seq)(raw_t) * jnp.ones_like(data["t_seq"])

                if raw_t < opt.prev_threshold:
                    prev = init_prev
                update, prev = model(params, subkey, data, prev)
                pos = update["pos"]
                data["pos"] = pos
                data["seq"] = jnp.argmax(update["aa"], axis=-1)
                maxprob = jax.nn.softmax(update["aa"], axis=-1).max(axis=-1)
                data["atom_pos"] = update["atom_pos"]
                data["aatype"] = update["aatype"]
                if step in out_steps:
                    results = [{key: value for key, value in data.items()}]
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
                            atom37_mask = mask[:, None] * get_atom37_mask(data["aatype"])
                            protein = Protein(np.array(atom37), np.array(data["aatype"]),
                                            np.array(atom37_mask), np.array(data["residue_index"]),
                                            np.array(chain), maxprob[:, None] * np.array(
                                                np.ones_like(atom37_mask, dtype=jnp.float32)))
                            pdb_string = to_pdb(protein)
                            with open(pdb_path, "wt") as f:
                                f.write(pdb_string)
                    if step == out_steps[-1]:
                        break
            print(f"Design {idx} generated in {time.time() - start:.3f} seconds.")
        print("All designs successfully generated.")
