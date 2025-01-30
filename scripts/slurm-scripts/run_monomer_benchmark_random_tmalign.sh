#!/bin/bash
#SBATCH -J salad-tmalign-single
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -p htc-el8
#SBATCH --mem 32G
#SBATCH -t 08:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

NOVOBENCH=/g/korbel/mjendrusch/repos/novobench/
config=$1
for num_aa in 50 100 200 300 400; do
  bash scripts/tmalign_success.sh ${config}_${num_aa}_random \
      "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-random" \
      "/g/korbel/mjendrusch/repos/benchmark_salad/monomers" &
  CUDA_VISIBLE_DEVICES="" python scripts/geometry_statistics.py \
      "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-random/" \
      "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/stats-${config}-${num_aa}-random.csv" &
done
wait