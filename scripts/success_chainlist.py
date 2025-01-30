import os
import shutil
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

input_path = sys.argv[1]
out_path = sys.argv[2]

# derived paths
bb_path = f"{input_path}/"
model = "esm"
score_path = f"{input_path}-{model}/scores.csv"
prediction_path = f"{input_path}-{model}/predictions/"
success_path = f"{input_path}-{model}/success"

with open(score_path, "rt") as f, open(out_path, "wt") as out_f:
    header = next(f)
    copied = set()
    for line in f:
        name, index, sequence, *numbers, ipae, mpae = line.strip().split(",")
        bb_index = int(name.split("_")[1])
        sc_rmsd, sc_tm, plddt, ptm, pae = map(float, numbers)
        success = (sc_rmsd < 2.0) and (plddt > (70.0 if model == "esm" else 80.0))
        backbone_name = f"{name}.pdb"
        if success and name not in copied:
            out_f.write(backbone_name + "\n")
            copied.add(name)