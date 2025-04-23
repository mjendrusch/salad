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

input_path = sys.argv[1]
package_path = sys.argv[2]
model = sys.argv[3]
genie = False
if len(sys.argv) >= 5:
    genie = sys.argv[4].strip() == "genie"

# derived paths
bb_path = f"{input_path}/"
score_path = f"{input_path}-{model}/scores.csv"
prediction_path = f"{input_path}-{model}/predictions/"
package_path = f"{package_path}-{model}/"
if genie:
    bb_path = f"{input_path}/pdbs/"
    score_path = f"{input_path}/{model}/scores.csv"
    prediction_path = f"{input_path}/{model}/predictions/"
    package_path = f"{sys.argv[2]}-{model}/"
# set up directory structure
os.makedirs(package_path, exist_ok=True)
os.makedirs(f"{package_path}/backbones/", exist_ok=True)
os.makedirs(f"{package_path}/predictions/", exist_ok=True)
# copy over scores
shutil.copyfile(score_path, f"{package_path}/scores.csv")

with open(score_path, "rt") as f:
    header = next(f)
    copied = dict()
    for line in f:
        name, index, sequence, *numbers, ipae, mpae = line.strip().split(",")
        bb_index = int(name.split("_")[1])
        sc_rmsd, sc_tm, plddt, ptm, pae = map(float, numbers)
        success = (sc_rmsd < 2.0) and (plddt > (70.0 if model == "esm" else 80.0))
        design_path = f"{prediction_path}/{name}/design_{index}.pdb"
        backbone_path = f"{bb_path}/{name}.pdb"
        # copy over successful backbones
        if success and name not in copied:
            copied[name] = []
            shutil.copyfile(backbone_path, f"{package_path}/backbones/{name}.pdb")
        if success:
            copied[name].append((design_path, f"{package_path}/predictions/{name}_{index}.pdb", sc_rmsd))
for name, items in copied.items():
    best = sorted(items, key=lambda x: x[-1])[0]
    shutil.copyfile(best[0], best[1])

