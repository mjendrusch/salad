import os
import numpy as np
from zipfile import BadZipFile

with open("chain_assemblies.csv", "wt") as f:
    base = "assembly_npz"
    result = {}
    for subdir in os.listdir(base):
        for name in os.listdir(f"{base}/{subdir}"):
            print(subdir, name)
            basename = name.split("-")[0].upper()
            full_name = f"{base}/{subdir}/{name}"
            try:
                chains = np.unique(np.load(full_name)["chain_index"])
            except KeyboardInterrupt as e:
                raise e
            except Exception:
                continue
            relevant_chains = [c for c in chains if "-" not in c]
            for chain in relevant_chains:
                combined = f"{basename}_{chain}"
                if combined not in result:
                    result[combined] = []
                result[combined].append(full_name)
    for combined in result:
        pdbid, chain = combined.split("_")
        for assembly in result[combined]:
            f.write(f"{pdbid},{chain},{assembly}\n")
            f.flush()