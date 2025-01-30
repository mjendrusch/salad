import os
import sys
import json

input_path = sys.argv[1]
output_path = sys.argv[2]
unit = int(sys.argv[3])
repeat = int(sys.argv[4])

all_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

result = dict()
for name in os.listdir(input_path):
    basename = name.split(".")[0]
    # tie all positions across chains
    tied_positions = [
        {all_chains[idy]: [idx + 1]  for idy in range(repeat)}
        for idx in range(unit)
    ]
    result[basename] = tied_positions
with open(output_path, "wt") as f:
    json.dump(result, f)
print(" ".join(all_chains[:repeat]))