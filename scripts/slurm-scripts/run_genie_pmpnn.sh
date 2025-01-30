#!/bin/bash
for size in 50 100 200 300 400 500; do
    sbatch scripts/run_pmpnn_ca.sh \
        /g/korbel/mjendrusch/repos/benchmark_salad/genie/monomers_${size}/pdbs/ \
        /g/korbel/mjendrusch/repos/benchmark_salad/genie/monomers_${size}/sequences/
done
