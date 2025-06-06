#!/bin/bash
#SBATCH -J salad-benchmark-single
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

config=$1
name=$2
path=$3
for num_aa in 50 100 200 300 400; do 
  python salad/training/eval_noise_schedule_benchmark.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa $num_aa --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/synthete/$name-${num_aa}-100s/" --num_steps 100 --out_steps "80" --prev_threshold 0.8 --num_designs 200 ${@:4}
  python salad/training/eval_noise_schedule_benchmark.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa $num_aa --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/synthete/$name-${num_aa}-200s/" --num_steps 200 --out_steps "180" --prev_threshold 0.8 --num_designs 200 ${@:4}
  python salad/training/eval_noise_schedule_benchmark.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa $num_aa --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/synthete/$name-${num_aa}-500s/" --num_steps 500 --out_steps "400" --prev_threshold 0.8 --num_designs 200 ${@:4}
done
