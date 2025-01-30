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
config=$1
for num_aa in 50 100 200 300 400 500 600 800 1000; do
  sbatch -C gpu=L40s --gres=gpu:L40s:1 -t 48:00:00 --dependency=afterok:16249246 \
    $NOVOBENCH/run_novobench.sh \
    --pdb-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-random/" \
    --fasta-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-random-sequences/seqs/" \
    --out-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-random-esm/"
done