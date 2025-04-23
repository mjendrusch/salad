import os
import sys
import json
import numpy as np
from salad.aflib.common.protein import from_pdb_string
from salad.aflib.common.residue_constants import atom_types
from salad.modules.utils.dssp import assign_dssp
import jax

AA_CODE = "ARNDCQEGHILKMFPSTWYVX"
def decode_sequence(sequence):
    sequence = np.array(sequence)
    result = []
    for idx in range(len(sequence)):
        aa = AA_CODE[sequence[idx]]
        result.append(aa)
    return "".join(result)
DSSP_CODE="LHE"
def decode_dssp(dssp):
    dssp = np.array(dssp)
    result = []
    for idx in range(len(dssp)):
        translated = DSSP_CODE[dssp[idx]]
        result.append(translated)
    return "".join(result)

def parse_chain(path):
    with open(path, "rt") as f:
        parent = from_pdb_string(f.read())
        N = parent.atom_positions[:, atom_types.index("N")]
        CA = parent.atom_positions[:, atom_types.index("CA")]
        C = parent.atom_positions[:, atom_types.index("C")]
        O = parent.atom_positions[:, atom_types.index("O")]
        ncaco = np.stack((N, CA, C, O), axis=1)
    std_ca = CA.std()
    dssp, *_ = assign_dssp(
        ncaco,
        np.zeros((CA.shape[0],), dtype=np.int32),
        np.ones((CA.shape[0],), dtype=np.bool_))
    dssp_string = decode_dssp(dssp)
    dssp_dist = np.array(jax.nn.one_hot(dssp, 3).mean(axis=0))
    return dssp_string, dssp_dist, std_ca

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(output_path, "wt") as f:
    f.write("name,dssp_string,loop,helix,strand,std\n")
    for name in os.listdir(input_path):
        if not name.endswith(".pdb"):
            continue
        basename = name.split(".")[0]
        dssp_string, dssp_dist, std = parse_chain(f"{input_path}/{name}")
        f.write(f"{basename},{dssp_string},{dssp_dist[0]},{dssp_dist[1]},{dssp_dist[2]},{std}\n")
        f.flush()
