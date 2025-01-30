#!/bin/bash
#SBATCH -J motif-benchmark
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 20:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

# single motif vp
# for target_name in `ls /g/korbel/mjendrusch/repos/genie2/data/design25/`; do
#   echo $target_name
#   target_path=/g/korbel/mjendrusch/repos/genie2/data/design25/$target_name
#   sbatch scripts/run_motif_benchmark_single_nocond.sh $target_path
# done

# multi motif vp
for target_name in `ls /g/korbel/mjendrusch/repos/genie2/data/multimotifs/`; do
  echo $target_name
  target_path=/g/korbel/mjendrusch/repos/genie2/data/multimotifs/$target_name
  sbatch scripts/run_motif_benchmark_single_nocond.sh $target_path
done