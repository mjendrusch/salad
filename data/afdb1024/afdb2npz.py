
if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    import numpy as np

    from salad.data.convert import parse_structure, read_components

    component_path = sys.argv[1]
    afdb_path = sys.argv[2]
    count = int(sys.argv[3])
    offset = int(sys.argv[4])
    good_csv = sys.argv[5]

    components = read_components(component_path)
    data = pd.read_csv(good_csv, sep=",")
    names = data["name"]

    for name in names[offset::count]:
        pdb = f"AF-{name}-F1-model_v4.pdb"
        base_name = f"AF-{name}-F1-model_v4"
        if not os.path.isfile(f"{afdb_path}/pdb/{pdb}"):
            continue
        if os.path.isfile(f"{afdb_path}/npz/{base_name}.npz") or os.path.isfile(f"wip_afdb_{base_name}"):
            continue
        else:
            with open(f"wip_afdb_{base_name}", "w") as f:
                f.write("wip")
        parse_result = parse_structure(components, f"{afdb_path}/pdb/{pdb}", get_bfactor=True)
        np.savez_compressed(f"{afdb_path}/npz/{base_name}.npz", **parse_result)
        os.remove(f"wip_afdb_{base_name}")
