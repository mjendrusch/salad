#!/bin/bash
#SBATCH -J confchange-benchmark
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 04:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

# autosym vp
config=default_vp
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_vp-1024-prod-nsb-default-vp-1/
pdb_path=/g/korbel/mjendrusch/repos/benchmark_salad/confchange/$config-parent-split
fasta_path=$pdb_path-sequences
bash scripts/run_pmpnn_confchange.sh $pdb_path $fasta_path 0.6
pdb_path=/g/korbel/mjendrusch/repos/benchmark_salad/confchange/$config-parent-split-constrained
fasta_path=$pdb_path-sequences
bash scripts/run_pmpnn_confchange.sh $pdb_path $fasta_path 0.5