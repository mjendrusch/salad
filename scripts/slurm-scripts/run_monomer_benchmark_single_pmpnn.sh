#!/bin/bash
#SBATCH -J salad-pmpnn-single
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 32:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

config=$1
for num_aa in 50 100 200 300 400; do 
  bash scripts/run_pmpnn.sh "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-100s/" "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-100s-sequences/"
  bash scripts/run_pmpnn.sh "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-200s/" "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-200s-sequences/"
  bash scripts/run_pmpnn.sh "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-500s/" "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-500s-sequences/"
done
