#!/bin/bash
#SBATCH -J salad-large-esm
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 48:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

NOVOBENCH=/g/korbel/mjendrusch/repos/novobench/
for num_aa in 50 100 200 300 400; do
  sbatch -C gpu=3090 --gres=gpu:3090:1 -t 48:00:00 \
    $NOVOBENCH/run_novobench.sh \
    --pdb-path "/g/korbel/mjendrusch/repos/benchmark_salad/genie/monomers_${num_aa}/pdbs/" \
    --fasta-path "/g/korbel/mjendrusch/repos/benchmark_salad/genie/monomers_${num_aa}/sequences/seqs/" \
    --out-path "/g/korbel/mjendrusch/repos/benchmark_salad/genie/monomers_${num_aa}/esm/"
done
for num_aa in 50 100 200 300 400; do
  sbatch -C gpu=3090 --gres=gpu:3090:1 -t 48:00:00 \
    $NOVOBENCH/run_novobench.sh \
    --pdb-path "/g/korbel/mjendrusch/repos/benchmark_salad/rfdiffusion/rfd-${num_aa}/" \
    --fasta-path "/g/korbel/mjendrusch/repos/benchmark_salad/rfdiffusion/rfd-${num_aa}-sequences/seqs/" \
    --out-path "/g/korbel/mjendrusch/repos/benchmark_salad/rfdiffusion/rfd-${num_aa}-esm/"
done