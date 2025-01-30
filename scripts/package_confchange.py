import os
import shutil
import sys
import json
import numpy as np
from alphafold.common.protein import from_pdb_string
from alphafold.common.residue_constants import atom_types

def parse_score(f):
    next(f)
    for line in f:
        name, index, sequence, *numbers, ipae, mpae = line.strip().split(",")
        sc_rmsd, sc_tm, plddt, ptm, pae = map(float, numbers)
        yield dict(name=name, index=int(index), sc_rmsd=sc_rmsd, plddt=plddt)

AA_CODE = "ARNDCQEGHILKMFPSTWYVX"
def decode_sequence(sequence):
    sequence = np.array(sequence)
    result = []
    for idx in range(len(sequence)):
        aa = AA_CODE[sequence[idx]]
        result.append(aa)
    return "".join(result)

input_path = sys.argv[1]
package_path = sys.argv[2] + "-af2"

# set up directory structure
os.makedirs(package_path, exist_ok=True)
for s, kind in enumerate(["parent", "child1", "child2"]):
    os.makedirs(f"{package_path}/backbones/{kind}/", exist_ok=True)
    os.makedirs(f"{package_path}/predictions/{kind}/", exist_ok=True)
    os.makedirs(f"{package_path}/backbones/partial_{kind}/", exist_ok=True)
    os.makedirs(f"{package_path}/predictions/partial_{kind}/", exist_ok=True)
    # copy over scores
    shutil.copyfile(f"{input_path}-af2-s{s}/scores.csv", f"{package_path}/scores_{kind}.csv")

with open(f"{input_path}-af2-s0/scores.csv", "rt") as fp, open(f"{input_path}-af2-s1/scores.csv", "rt") as fc1, open(f"{input_path}-af2-s2/scores.csv", "rt") as fc2:
    copied = dict()
    for item in zip(parse_score(fp), parse_score(fc1), parse_score(fc2)):
        success = item[0]["index"] != 0
        successes = []
        for idx, ii in enumerate(item):
            success = success and (ii["sc_rmsd"] < 3.0) and (ii["plddt"] > 75)
            successes.append(success)
        partial_success = (not success) and successes[0] and (successes[1] or successes[2])
        name = "_".join(item[0]["name"].split("_")[:-1])
        index = item[0]["index"]
        # copy over successful backbones
        if success:
            for s, kind in enumerate(["parent", "child1", "child2"]):
                prediction_path = f"{input_path}-af2-s{s}/predictions/"
                design_path = f"{prediction_path}/{name}_s{s}/design_{index}.pdb"
                backbone_path = f"{input_path}/{name}_s{s}.pdb"
                shutil.copyfile(backbone_path, f"{package_path}/backbones/{kind}/{name}_{kind}.pdb")
                shutil.copyfile(design_path, f"{package_path}/predictions/{kind}/{name}_{kind}_{index}.pdb")
        elif partial_success:
            for s, kind in enumerate(["parent", "child1", "child2"]):
                prediction_path = f"{input_path}-af2-s{s}/predictions/"
                design_path = f"{prediction_path}/{name}_s{s}/design_{index}.pdb"
                backbone_path = f"{input_path}/{name}_s{s}.pdb"
                shutil.copyfile(backbone_path, f"{package_path}/backbones/partial_{kind}/{name}_{kind}.pdb")
                shutil.copyfile(design_path, f"{package_path}/predictions/partial_{kind}/{name}_{kind}_{index}.pdb")
