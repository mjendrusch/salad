import os
import sys
import json
import Bio
import Bio.PDB
import Bio.PDB.PDBExceptions
import numpy as np
from alphafold.common.protein import from_pdb_string
from alphafold.common.residue_constants import atom_types


AA_CODE = "ARNDCQEGHILKMFPSTWYVX"
def decode_sequence(sequence):
    sequence = np.array(sequence)
    result = []
    for idx in range(len(sequence)):
        aa = AA_CODE[sequence[idx]]
        result.append(aa)
    return "".join(result)

def parse_chain(path, chain="A", offset=0.0):
    offset = offset * 100.0 * np.array([1.0, 1.0, 1.0], dtype=np.float32)
    with open(path, "rt") as f:
        parent = from_pdb_string(f.read())
        seq = decode_sequence(parent.aatype)
        N = parent.atom_positions[:, atom_types.index("N")] + offset
        CA = parent.atom_positions[:, atom_types.index("CA")] + offset
        C = parent.atom_positions[:, atom_types.index("C")] + offset
        O = parent.atom_positions[:, atom_types.index("O")] + offset
        chain_info = {
            f"seq_chain_{chain}": seq,
            f"coords_chain_{chain}": {
                f"N_chain_{chain}": N.tolist(),
                f"CA_chain_{chain}": CA.tolist(),
                f"C_chain_{chain}": C.tolist(),
                f"O_chain_{chain}": O.tolist(),
            }
        }
    return chain_info, len(seq)

def splice_seq(constring, motif_seq, seq):
    result = []
    for c, m, s in zip(constring, motif_seq, seq):
        if c == "X":
            result.append(s)
        else:
            result.append(m)
    return "".join(result)

def fix_motif(constring):
    result = []
    for idx, c in enumerate(constring):
        if c == "F":
            result.append(idx + 1)
    return result

def expand_motif(seq, constring, consubset):
    subseq = [
        s
        for c, s in zip(consubset, seq)
        if c == "F"
    ]
    result = []
    for c in constring:
        if c == "X":
            result.append("X")
        else:
            next_aa, *subseq = subseq
            result.append(next_aa)
    return "".join(result)

def parse_motif(path):
    segments = []
    motif_groups = []
    with open(path) as f:
        for line in f:
            line = line.strip()
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
                    segments.append(((end - start + 1) * "F", motif_group))
    motif_groups = list(set(motif_groups))
    consubsets = {m: "" for m in motif_groups}
    for segval, motif_group in segments:
        for m in motif_groups:
            if m == motif_group:
                consubsets[m] += segval
                for mm in motif_groups:
                    if mm != m:
                        consubsets[mm] += "X" * len(segval)
    with open(path, "rt") as f:
        protein = from_pdb_string(f.read())
    motif_sequence = decode_sequence(protein.aatype)
    xyz = protein.atom_positions
    motif_ca = xyz[:, atom_types.index("CA")]
    return motif_groups, consubsets, motif_sequence, motif_ca

input_path = sys.argv[1]
motif_path = sys.argv[2]
# motif_groups, consubsets, motif_seq, motif_ca = parse_motif(motif_path)

out_path = sys.argv[3]
os.makedirs(out_path, exist_ok=True)
fixed_path = sys.argv[4]
task_path = sys.argv[5]
multiplicity = 1
if len(sys.argv) > 6:
    multiplicity = int(sys.argv[6])

all_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

fixed_dict = dict()
parsed_tasks = []
# load constrings
constrings = []
with open(f"{input_path}/constrings.jsonl", "rt") as f:
    for line in f:
        # if we pass multiplicity, add the same
        # constring N times. This is to accomodate
        # runs with structures taken at multiple time points
        # for the same constring
        for _ in range(multiplicity):
            constrings.append(json.loads(line))
# sort PDB files by their design index
names = [n for n in os.listdir(input_path) if n.endswith(".pdb")]
names = sorted(names, key=lambda x: int(x.split("_")[1]))
for name, conspec in zip(names, constrings):
    basename = ".".join(name.split(".")[:-1])
    try:
        task_info, len_design = parse_chain(
            f"{input_path}/{name}", chain="A", offset=0.0)
    except Bio.PDB.PDBExceptions.PDBConstructionException:
        print("Warning: skipped bad PDB file:", basename, file=sys.stderr)
        continue
    fixed_positions = []
    for motif_group, constring in conspec.items():
        fixed_positions += fix_motif(constring)
        # NOTE: we do not need this, as the sequence
        # is already fixed at generation time!
        #
        # task_info["seq_chain_A"] = splice_seq(
        #     constring,
        #     expand_motif(motif_seq, constring, consubsets[motif_group]),
        #     task_info["seq_chain_A"])
    task_info["name"] = basename
    task_info["num_of_chains"] = 1
    task_info["seq"] = task_info["seq_chain_A"]
    parsed_tasks.append(task_info)
    fixed_dict[basename] = {"A": fixed_positions}
with open(fixed_path, "wt") as f:
    json.dump(fixed_dict, f)
with open(task_path, "wt") as f:
    for item in parsed_tasks:
        json.dump(item, f)
        f.write("\n")