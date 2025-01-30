#!/bin/bash
#SBATCH -J train-sae
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p gpu-el8
#SBATCH -C gpu=3090
#SBATCH --gres=gpu:3090:8
#SBATCH --mem-per-gpu 64283
#SBATCH -t 48:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
module load CUDA/12.0.0
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

python salad/training/train_structure_autoencoder.py \
  --path      /g/korbel/mjendrusch/runs/debug-salad-new/ \
  --data_path /g/korbel/mjendrusch/data/ \
  $@
