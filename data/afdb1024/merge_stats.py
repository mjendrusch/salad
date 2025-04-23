import sys
import pandas as pd

out_path = sys.argv[1]
count = int(sys.argv[2])

data = pd.concat([
    pd.read_csv(f"{out_path}.{i}", sep=",")
    for i in range(count)
])
data.to_csv(out_path, sep=",", header=True, index=False)
