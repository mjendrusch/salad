#!/bin/bash
for size in 300 400 500; do
    sbatch scripts/run_rfd.sh $size
done