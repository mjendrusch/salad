#!/bin/bash
#SBATCH -J salad-pmpnn-letter
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 24:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

for letter in S A L D; do 
  bash scripts/run_pmpnn.sh "/g/korbel/mjendrusch/repos/benchmark_salad/shape/ve-seg-${letter}-1/" "/g/korbel/mjendrusch/repos/benchmark_salad/shape/ve-seg-${letter}-1-sequences/"
done
