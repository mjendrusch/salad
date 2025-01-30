import os
import sys

in_path = sys.argv[1]
out_path = sys.argv[2]
filenames = os.listdir(in_path)
names = set()
out_names = []
for filename in filenames:
    bb_name = "_".join(filename.split("_")[:3])
    if bb_name not in names and bb_name.endswith("400"):
        names.add(bb_name)
        out_names.append(filename)
with open(out_path, "wt") as out_f:
    for filename in out_names:
        out_f.write(filename + "\n")
