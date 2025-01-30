#!/bin/bash
for size in 50 100 200 300 400 500; do
    sbatch scripts/run_pmpnn.sh \
        /g/korbel/mjendrusch/repos/benchmark_salad/rfdiffusion/rfd-${size}/ \
        /g/korbel/mjendrusch/repos/benchmark_salad/rfdiffusion/rfd-${size}-sequences/
done
