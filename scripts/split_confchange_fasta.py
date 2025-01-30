import os
import sys

path = sys.argv[1]
for name in os.listdir(path):
    basename = name.split(".")[0]
    full_path = f"{path}/{name}"
    with open(full_path) as f, \
         open(f"{path}/{basename}_s0.fasta", "wt") as pf, \
         open(f"{path}/{basename}_s1.fasta", "wt") as c1f, \
         open(f"{path}/{basename}_s2.fasta", "wt") as c2f:
        while True:
            try:
                header = next(f).strip()
                sequences = next(f).strip()
                p, c1, c2 = sequences.split("/")
                pf.write(header + "\n")
                pf.write(p + "\n")
                c1f.write(header + "\n")
                c1f.write(c1 + "\n")
                c2f.write(header + "\n")
                c2f.write(c2 + "\n")
            except StopIteration:
                break
