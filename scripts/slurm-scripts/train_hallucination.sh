#!/bin/bash
#SBATCH -J train-hallucination
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --mem-per-gpu 64283
#SBATCH -p gpu-el8
#SBATCH --gres=gpu:3090:8
#SBATCH -t 24:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
module load CUDA/11.3.1
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

python salad/training/train_hallucination.py \
  --path      /g/korbel/mjendrusch/runs/debug-salad-new/ \
  --data_path /g/korbel/mjendrusch/data/ \
  $@
