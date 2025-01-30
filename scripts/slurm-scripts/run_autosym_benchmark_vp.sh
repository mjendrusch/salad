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

# autosym vp
config=default_vp
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_vp-1024-prod-nsb-default-vp-1/
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 100 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-2-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --replicate_count 2 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-3-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --replicate_count 3 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 200 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-4-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --replicate_count 4 --autosym True
# complex
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa "100:100" \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-autosym-cpx-2-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --replicate_count 2 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa "50:50" \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-2-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --replicate_count 2 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa "50:50:50" \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-3-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --replicate_count 3 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa "50:50:50:50" \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-4-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --replicate_count 4 --autosym True

# cyclic monomer VE
config=default_ve_minimal_timeless
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_ve_minimal_timeless-1024-prod-nsb-default-ve-mtl-1/
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 100 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-2-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --replicate_count 2 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-3-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --replicate_count 3 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 200 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-4-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --replicate_count 4 --autosym True
# complex
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa "100:100" \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-autosym-cpx-2-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --replicate_count 2 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa "50:50" \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-2-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --replicate_count 2 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa "50:50:50" \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-3-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --replicate_count 3 --autosym True
python salad/training/eval_repeat.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa "50:50:50:50" \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-4-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --replicate_count 4 --autosym True
