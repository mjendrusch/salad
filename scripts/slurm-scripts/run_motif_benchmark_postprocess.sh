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

for basepath in "/g/korbel/mjendrusch/repos/benchmark_salad/motif/cond/multimotif_vp" "/g/korbel/mjendrusch/repos/benchmark_salad/motif/nocond/default_vp"; do
# single motif vp
for target_name in `ls /g/korbel/mjendrusch/repos/genie2/data/design25/`; do
  echo $target_name
  target_path=/g/korbel/mjendrusch/repos/genie2/data/design25/$target_name
  pdb_path=${basepath}-${target_name}
  python scripts/motif_success.py $pdb_path $target_path
done

# multi motif vp
for target_name in `ls /g/korbel/mjendrusch/repos/genie2/data/multimotifs/`; do
  echo $target_name
  target_path=/g/korbel/mjendrusch/repos/genie2/data/multimotifs/$target_name
  pdb_path=${basepath}-${target_name}
  python scripts/motif_success.py $pdb_path $target_path
done
done