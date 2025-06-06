import os
import sys
import json

input_path = sys.argv[1]
output_path = sys.argv[2]
unit = int(sys.argv[3])
repeat = int(sys.argv[4])

result = dict()
for name in os.listdir(input_path):
    basename = name.split(".")[0]
    tied_positions = [
        {"A": [idx + idy * unit + 1 for idy in range(repeat)]}
        for idx in range(unit)
    ]
    result[basename] = tied_positions
with open(output_path, "wt") as f:
    json.dump(result, f)