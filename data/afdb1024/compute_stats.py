"""Script to compute AlphaFold DB structure statistics.

This script is designed to be run as multiple parallel processes.
Specify a number of processes and the offset within that array of those processes
for each instance.
The script will split up the AlphaFold DB dataset and write one part file
per process. Run merge_stats.py to merge the resulting files.
"""

import os
import pydssp
import sys
import gzip
import pandas as pd
import numpy as np
import tqdm

from salad.aflib.common.protein import from_pdb_string
from salad.aflib.common.residue_constants import atom_types

# base directory of the dataset
base = sys.argv[1]
# name of the resulting statistics file
out_path = sys.argv[2]
# number of parallel processes
count = int(sys.argv[3])
# offset in the array of parallel processes
offset = int(sys.argv[4])

with gzip.open(f"{base}/2-repId_isDark_nMem_repLen_avgLen_repPlddt_avgPlddt_LCAtaxId.tsv.gz", "rt") as f:
    afdb_stats = pd.read_csv(f, sep="\t", header=None, usecols=[0, 3, 5], names=["name", "length", "plddt"])
afdb1024 = afdb_stats[afdb_stats["length"] <= 1024]

with open(out_path + f".{offset}", "wt") as out_f:
    out_f.write("name,length,plddt,L,H,E,std_ca\n")
    tick = 0
    # for each structure in the dataset, extract length, pLDDT and secondary structure
    for name, length, plddt in tqdm.tqdm(zip(
            afdb1024["name"][offset::count],
            afdb1024["length"][offset::count],
            afdb1024["plddt"][offset::count])):
        pdb_path = f"{base}/pdb/AF-{name}-F1-model_v4.pdb"
        if not os.path.isfile(pdb_path):
            continue
        with open(pdb_path, "rt") as f:
            protein = from_pdb_string(f.read())
        N = protein.atom_positions[:, atom_types.index("N")]
        CA = protein.atom_positions[:, atom_types.index("CA")]
        C = protein.atom_positions[:, atom_types.index("C")]
        O = protein.atom_positions[:, atom_types.index("O")]
        std_ca = CA.std()
        bb = np.stack((N, CA, C, O), axis=1)
        L, H, E = pydssp.main.assign(bb, out_type="onehot").astype(np.float32).mean(axis=0)
        out_f.write(f"{name},{length},{plddt:.2f},{L:.2f},{H:.2f},{E:.2f},{std_ca:.2f}\n")
        if tick % 100 == 0:
            out_f.flush()
        tick += 1
