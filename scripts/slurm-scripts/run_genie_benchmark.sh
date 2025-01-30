#!/bin/bash
for size in 50 100 200 300 400 500; do
    sbatch scripts/run_genie.sh $size
done