"""Script to concatenate statistics from multiple parallel runs of compute_stats.py"""

import sys
import pandas as pd

# base name of the output statistics file
out_path = sys.argv[1]
# number of parallel processes used to produce statistics files
count = int(sys.argv[2])

data = pd.concat([
    pd.read_csv(f"{out_path}.{i}", sep=",")
    for i in range(count)
])
data.to_csv(out_path, sep=",", header=True, index=False)
