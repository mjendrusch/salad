import os
import shutil
import sys
import json
import numpy as np
from salad.aflib.common.protein import from_pdb_string
from salad.aflib.common.residue_constants import atom_types


AA_CODE = "ARNDCQEGHILKMFPSTWYVX"
def decode_sequence(sequence):
    sequence = np.array(sequence)
    result = []
    for idx in range(len(sequence)):
        aa = AA_CODE[sequence[idx]]
        result.append(aa)
    return "".join(result)

def parse_bb(path):
    with open(path, "rt") as f:
        parent = from_pdb_string(f.read())
        seq = decode_sequence(parent.aatype)
        N = parent.atom_positions[:, atom_types.index("N")]
        CA = parent.atom_positions[:, atom_types.index("CA")]
        C = parent.atom_positions[:, atom_types.index("C")]
        O = parent.atom_positions[:, atom_types.index("O")]
        bb = np.stack((N, CA, C, O), axis=1)
        resi = parent.residue_index
        chain = parent.chain_index
    return bb, resi, chain, len(seq)

def align_motif(motif_bb, bb, constring):
    motif_ca_gt_flat = []
    motif_ca_flat = []
    motif_bb_gt_flat = []
    motif_bb_flat = []
    for m, cs in constring.items():
        conmask = np.array([c == "F" for c in cs], dtype=np.bool_)
        mca_gt = motif_bb[m][:, 1]
        mbb_gt = motif_bb[m].reshape(-1, 3)
        mca = bb[conmask][:, 1]
        mbb = bb[conmask].reshape(-1, 3)
        mca = kabsch(np.copy(mca), np.copy(mca_gt))
        mbb = kabsch(np.copy(mbb), np.copy(mbb_gt))
        motif_bb_gt_flat.append(mbb_gt)
        motif_bb_flat.append(mbb)
        motif_ca_gt_flat.append(mca_gt)
        motif_ca_flat.append(mca)
    motif_bb_gt_flat = np.concatenate(motif_bb_gt_flat, axis=0)
    motif_bb_flat = np.concatenate(motif_bb_flat, axis=0)
    motif_ca_gt_flat = np.concatenate(motif_ca_gt_flat, axis=0)
    motif_ca_flat = np.concatenate(motif_ca_flat, axis=0)
    motif_rmsd_ca = np.sqrt(((motif_ca_flat - motif_ca_gt_flat) ** 2).sum(axis=-1).mean())
    motif_rmsd_bb = np.sqrt(((motif_bb_flat - motif_bb_gt_flat) ** 2).sum(axis=-1).mean())
    return motif_rmsd_ca, motif_rmsd_bb

def alignment_parameters(x, y):
    x_center = x.mean(axis=0)
    y_center = y.mean(axis=0)
    x = x - x_center
    y = y - y_center
    covariance = (x[:, :, None] * y[:, None, :]).sum(axis=0)
    u, _, v = np.linalg.svd(
        covariance, compute_uv=True, full_matrices=True)
    det_sign = np.linalg.det(u) * np.linalg.det(v)
    det_flip = np.ones((3,))
    det_flip[-1] = det_sign
    rot = np.einsum("ak,kb->ba", u * det_flip[None, :], v)
    return rot, x_center, y_center

def kabsch(x, y):
    rot, x_center, y_center = alignment_parameters(x, y)
    result = np.einsum(
        "ak,ik->ia", rot, (x - x_center[None])) + y_center[None]
    return result

def parse_constrings(constring_path, bb_path):
    constring_list = []
    with open(constring_path, "rt") as f:
        for line in f:
            item = json.loads(line.strip())
            constring_list.append(item)
    all_pdbs = list(set([
        int(name.split("_")[1])
        for name in os.listdir(bb_path)
        if name.endswith(".pdb")]))
    all_pdbs = sorted(all_pdbs)
    constring_dict = dict()
    for cs, index in zip(constring_list, all_pdbs):
        constring_dict[index] = cs
    return constring_dict

def get_consubsets(motif_groups, segments):
    consubsets = {m: "" for m in motif_groups}
    total_aa = 0
    for (start, end), motif_group in segments:
        segval = (end - start + 1) * "F"
        for m in motif_groups:
            if m == motif_group:
                total_aa += len(segval)
                consubsets[m] += segval
                for mm in motif_groups:
                    if mm != m:
                        consubsets[mm] += "X" * len(segval)
    return consubsets

def segment_rearrange(x, resi, chain, segment_range_chain):
    chain_order = np.unique([c[-1] for c in segment_range_chain])
    result = []
    for rng, cid in segment_range_chain:
        cid = np.argmax(chain_order == cid)
        mask = (cid == chain) * (resi[:, None] == rng[None, :]).any(axis=1)
        result.append(x[mask])
    return np.concatenate(result, axis=0)

def parse_motif_spec(x):
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
            if chain_name != " ":
                segments.append(((start, end), motif_group))
                segment_range_chain.append((np.arange(start, end + 1, dtype=np.int32), chain_name))
    motif_groups = list(set(motif_groups))
    return segment_range_chain, segments, motif_groups

def prepare_motif_bb(motif_path):
    with open(motif_path, "rt") as f:
        segment_range_chain, segments, motif_groups = parse_motif_spec(f.read())
    consubsets = get_consubsets(motif_groups, segments)
    ca, resi, chain, _ = parse_bb(motif_path)
    ca = segment_rearrange(ca, resi, chain, segment_range_chain)
    result = dict()
    for motif_group in motif_groups:
        subset_mask = np.array([
            c == "F" for c in consubsets[motif_group]], dtype=np.bool_)
        result[motif_group] = ca[subset_mask]
    return result

input_path = sys.argv[1]
motif_path = sys.argv[2]

# derived paths
constring_path = f"{input_path}/constrings.jsonl"
bb_path = f"{input_path}/"
score_path = f"{input_path}-esm/scores.csv"
prediction_path = f"{input_path}-esm/predictions/"
out_score_path = f"{input_path}-esm/motif_scores.csv"
success_path = f"{input_path}-esm/success"
os.makedirs(success_path + "_ca", exist_ok=True)
os.makedirs(success_path + "_bb", exist_ok=True)

all_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

motif_bb = prepare_motif_bb(motif_path)
constrings = parse_constrings(constring_path, bb_path)
with open(score_path, "rt") as f, open(out_score_path, "wt") as out_f:
    header = next(f)
    out_f.write("name,index,sequence,motif_rmsd_ca,motif_rmsd_bb,sc_rmsd,sc_tm,plddt,ptm,pae,ipae,mpae\n")
    for line in f:
        name, index, sequence, *numbers, ipae, mpae = line.strip().split(",")
        bb_index = int(name.split("_")[1])
        constring = constrings[bb_index]
        sc_rmsd, sc_tm, plddt, ptm, pae = map(float, numbers)
        success = (sc_rmsd < 2.0) and (plddt > 70.0)
        design_path = f"{prediction_path}/{name}/design_{index}.pdb"
        backbone_path = f"{bb_path}/{name}.pdb"
        design_bb, _, _, _ = parse_bb(design_path)
        motif_rmsd_ca, motif_rmsd_bb = align_motif(motif_bb, design_bb, constring)
        out_f.write(",".join([name, index, sequence, str(motif_rmsd_ca), str(motif_rmsd_bb)] + numbers + [ipae, mpae])+ "\n")
        out_f.flush()
        success_ca = success and (motif_rmsd_ca <= 1.0)
        success_bb = success and (motif_rmsd_bb <= 1.0)
        if success_ca:
            shutil.copyfile(design_path, f"{success_path}_ca/{name}_{index}.pdb")
        if success_bb:
            shutil.copyfile(design_path, f"{success_path}_bb/{name}_{index}.pdb")
