#!/bin/bash
#SBATCH -J train-latent-diff
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64GB
#SBATCH -p gpu-el8
#SBATCH --gpus 8
#SBATCH -t 48:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
module load CUDA/11.3.1
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

python salad/training/train_simple_diffusion.py \
  --path      /g/korbel/mjendrusch/runs/debug-salad-new/ \
  --data_path /g/korbel/mjendrusch/data/ \
  $@
