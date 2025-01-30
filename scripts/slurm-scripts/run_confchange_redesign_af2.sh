#!/bin/bash
#SBATCH -J confchange-af2
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
AF2PATH=/g/korbel/mjendrusch/repos/alpha-design/
NOVOBENCH=/g/korbel/mjendrusch/repos/novobench/
# autosym vp
config=default_vp_omap
pdb_path=/g/korbel/mjendrusch/repos/benchmark_salad/confchange/$config-parent-split-almost-hits
fasta_path=$pdb_path-sequences/seqs/
out_path=$pdb_path-af2
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path-s0/ --filter-name "_s0"
# CUDA_VISIBLE_DEVICES=1 bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
#     --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path-s1/ --filter-name "_s1" &
# CUDA_VISIBLE_DEVICES=2 bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
#     --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path-s2/ --filter-name "_s2" &
# wait