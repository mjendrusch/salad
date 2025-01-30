#!/bin/bash
#SBATCH -J salad-genie-bench
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=3090
#SBATCH --gres=gpu:3090:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 12:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source ~/.bashrc
conda activate genie

OUT_DIR=/g/korbel/mjendrusch/repos/benchmark_salad/genie/
count=$1
cd /g/korbel/mjendrusch/repos/genie2/
python genie/sample_unconditional.py --name base --epoch 40 --scale 0.6  \
                                     --rootdir /g/korbel/mjendrusch/repos/denovome/genie_params/             \
                                     --outdir $OUT_DIR/monomers_$count/ \
                                     --min_length $count --max_length $count    \
                                     --num_samples 200 --batch_size 1 > $OUT_DIR/${count}aa_log