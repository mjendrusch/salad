#!/bin/bash
#SBATCH -J salad-rfd-bench
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p gpu-el8
#SBATCH --gres=gpu:3090:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 12:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
module load CUDA/12.0.0
conda activate SE3nv

size=$1
OUT_PATH=/g/korbel/mjendrusch/repos/benchmark_salad/rfdiffusion/
PROTEINMPNN=/g/korbel/mjendrusch/repos/rf_diffusion_env/ProteinMPNN/
cd /g/korbel/mjendrusch/repos/rf_diffusion_env/RFdiffusion
python scripts/run_inference.py "contigmap.contigs=[$size-$size]" inference.output_prefix=$OUT_PATH/rfd-$size/result inference.num_designs=200 > ${size}aa_log