#!/bin/bash
#SBATCH -J motif-benchmark-postprocess
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 12:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad
AF2PATH=/g/korbel/mjendrusch/repos/alpha-design/
NOVOBENCH=/g/korbel/mjendrusch/repos/novobench/

# a total of 9 * 5 * 3 * 2 = 270 directories of 200 * 11 designs each (2,200 * 270 = 594,000 sequences)
basepath="/g/korbel/mjendrusch/repos/benchmark_salad/monomers/"
for config in default_vp default_vp_scaled default_ve_scaled default_vp_timeless default_vp_scaled_timeless default_ve_timeless \
              default_vp_minimal_timeless default_vp_scaled_minimal_timeless default_ve_minimal_timeless; do
  for size in 50 100 200 300 400; do
    for steps in 100 200 500; do
      input_path=${basepath}/${config}-${size}-${steps}s
      python scripts/success.py $input_path
    done
  done
done

# a total of 3 * 5 = 15 directories (15 * 200 * 11 = 33,000 sequences)
for config in default_vp default_vp_scaled default_ve_scaled; do
  for size in 50 100 200 300 400; do
    input_path=${basepath}/${config}-${size}-random
    python scripts/success.py $input_path
  done
done

