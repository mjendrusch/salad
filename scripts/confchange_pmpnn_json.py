import os
import sys
import json
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

input_path = sys.argv[1]
tied_path = sys.argv[2]
task_path = sys.argv[3]
weight = float(sys.argv[4])

all_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

tied_dict = dict()
parsed_tasks = []
for name in os.listdir(input_path):
    if not "_s0" in name:
        continue
    parent = name
    basename = name.split("_s0")[0]
    child_1 = parent.split("_s0")[0] + "_s1.pdb"
    child_2 = parent.split("_s0")[0] + "_s2.pdb"
    task_info = dict()
    parent, len_parent = parse_chain(
        f"{input_path}/{parent}", chain="A",
        offset=0.0)
    task_info.update(parent)
    child_1, len_child_1 = parse_chain(
        f"{input_path}/{child_1}", chain="B",
        offset=5.0)
    task_info.update(child_1)
    child_2, len_child_2 = parse_chain(
        f"{input_path}/{child_2}", chain="C",
        offset=-5.0)
    task_info.update(child_2)
    task_info["name"] = basename
    task_info["num_of_chains"] = 3
    task_info["seq"] = task_info["seq_chain_A"] + task_info["seq_chain_B"] + task_info["seq_chain_C"]
    parsed_tasks.append(task_info)
    # tie all positions across chains
    tied_positions = []
    for idx in range(len_parent):
        if idx < len_child_1:
            tied_positions.append({"A": [[idx + 1], [weight]], "B": [[idx + 1], [1 - weight]]})
        else:
            tied_positions.append({"A": [[idx + 1], [weight]], "C": [[idx - len_child_1 + 1], [1 - weight]]})
    tied_dict[basename] = tied_positions
with open(tied_path, "wt") as f:
    json.dump(tied_dict, f)
with open(task_path, "wt") as f:
    for item in parsed_tasks:
        json.dump(item, f)
        f.write("\n")