import os
import shutil
import sys

input_path = sys.argv[1]
package_path = sys.argv[2]

# derived paths
constring_path = f"{input_path}/constrings.jsonl"
bb_path = f"{input_path}/"
score_path = f"{input_path}-esm/scores.csv"
prediction_path = f"{input_path}-esm/predictions/"
out_score_path = f"{input_path}-esm/motif_scores.csv"
success_path = f"{input_path}-esm/success_bb/"
package_path = f"{package_path}-esm/"
# make output paths
os.makedirs(package_path + "backbones/", exist_ok=True)
os.makedirs(package_path + "predictions/", exist_ok=True)

# copy over score file
shutil.copyfile(out_score_path, f"{package_path}/motif_scores.csv")

# copy over unique successes
with open(out_score_path, "rt") as f:
    header = next(f)
    copied = dict()
    for line in f:
        name, index, sequence, *numbers, ipae, mpae = line.strip().split(",")
        bb_index = int(name.split("_")[1])
        bb, ca, sc_rmsd, sc_tm, plddt, ptm, pae = map(float, numbers)
        success = (bb <= 1.0) and (sc_rmsd < 2.0) and (plddt > 70.0)
        design_path = f"{prediction_path}/{name}/design_{index}.pdb"
        backbone_path = f"{bb_path}/{name}.pdb"
        if success and (name not in copied):
            copied[name] = []
            shutil.copyfile(backbone_path, f"{package_path}/backbones/{name}.pdb")
        if success:
            copied[name].append((design_path, f"{package_path}/predictions/{name}_{index}.pdb", sc_rmsd))
    for name, items in copied.items():
        design = sorted(items, key=lambda x: x[-1])[0]
        shutil.copyfile(design[0], design[1])