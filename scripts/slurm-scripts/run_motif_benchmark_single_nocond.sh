#!/bin/bash
#SBATCH -J single-motif-single
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 5:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

# nocond motif vp
target_path=$1
target=$(basename $target_path)
# config=default_vp
# path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_vp-1024-prod-nsb-default-vp-1/
# python /g/korbel/mjendrusch/repos/salad/salad/training/eval_motif_benchmark_nocond.py --config $config \
#   --params "$path/checkpoint-ema-200000.jax" \
#   --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/motif/nocond/$config-$target" \
#   --num_steps 500 --out_steps "400,499" --prev_threshold 0.8 --num_designs 1000 --timescale_pos "cosine(t)" \
#   --template $target_path

# conditional
config=multimotif_vp
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-multimotif_vp-1024-debug-nsb-multimotif-2/
python /g/korbel/mjendrusch/repos/salad/salad/training/eval_motif_benchmark.py --config $config \
  --params "$path/checkpoint-ema-200000.jax" \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/motif/cond/$config-$target" \
  --num_steps 500 --out_steps "400,499" --prev_threshold 0.8 --num_designs 1000 --timescale_pos "cosine(t)" \
  --template $target_path
# config=default_vp_scaled
# path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_vp_scaled-1024-prod-nsb-default-vp-scaled-1/
# python /g/korbel/mjendrusch/repos/salad/salad/training/eval_motif_benchmark_nocond.py --config $config \
#   --params "$path/checkpoint-ema-200000.jax" \
#   --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/motif/nocond/$config-$target" \
#   --num_steps 500 --out_steps "400" --prev_threshold 0.8 --num_designs 1000 --timescale_pos "cosine(t)" --cloud_std "default(num_aa)" \
#   --template $target_path
