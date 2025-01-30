#!/bin/bash
#SBATCH -J salad-benchmark-single
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 32:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

# vp screw
config=default_vp
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_vp-1024-prod-nsb-default-vp-1/
# VP-screw
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-S3-a120-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 120.0 \
  --screw_translation 10.0 --replicate_count 3 --mode "rotation" --mix_output couple
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-S3-a60-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 60.0 \
  --screw_translation 10.0 --replicate_count 3 --mode "rotation" --mix_output couple
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-S3-a30-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 30.0 \
  --screw_translation 10.0 --replicate_count 3 --mode "rotation" --mix_output couple

# cyclic monomer VE
config=default_ve_minimal_timeless
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_ve_minimal_timeless-1024-prod-nsb-default-ve-mtl-1/
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C3-r8-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 8.0 --screw_angle 120.0 \
  --screw_translation 0.0 --replicate_count 3 --mode "screw" --mix_output couple --sym_noise True
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C3-r10-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 10.0 --screw_angle 120.0 \
  --screw_translation 0.0 --replicate_count 3 --mode "screw" --mix_output couple --sym_noise True
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C3-r12-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 12.0 --screw_angle 120.0 \
  --screw_translation 0.0 --replicate_count 3 --mode "screw" --mix_output couple --sym_noise True
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 200 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C4-r10-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 10.0 --screw_angle 90.0 \
  --screw_translation 0.0 --replicate_count 4 --mode "screw" --mix_output couple --sym_noise True
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 200 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C4-r12-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 12.0 --screw_angle 90.0 \
  --screw_translation 0.0 --replicate_count 4 --mode "screw" --mix_output couple --sym_noise True
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 250 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C5-r12-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 12.0 --screw_angle 72.0 \
  --screw_translation 0.0 --replicate_count 5 --mode "screw" --mix_output couple --sym_noise True
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 250 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C5-r14-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 14.0 --screw_angle 72.0 \
  --screw_translation 0.0 --replicate_count 5 --mode "screw" --mix_output couple --sym_noise True
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 300 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C6-r12-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 12.0 --screw_angle 60.0 \
  --screw_translation 0.0 --replicate_count 6 --mode "screw" --mix_output couple --sym_noise True
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 300 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C6-r14-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 14.0 --screw_angle 60.0 \
  --screw_translation 0.0 --replicate_count 6 --mode "screw" --mix_output couple --sym_noise True
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 350 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C7-r14-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 20 --timescale_pos "ve(t)" --screw_radius 14.0 --screw_angle 51.42 \
  --screw_translation 0.0 --replicate_count 7 --mode "screw" --mix_output couple --sym_noise True

config=default_vp
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_vp-1024-prod-nsb-default-vp-1/
# cyclic-monomer
# python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 100 \
#   --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-C2-1" --num_steps 500 --out_steps "400" \
#   --prev_threshold 0.8 --num_designs 50 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 180.0 \
#   --screw_translation 0.0 --replicate_count 2 --mode "rotation" --mix_output couple
# python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 200 \
#   --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-C2-1" --num_steps 500 --out_steps "400" \
#   --prev_threshold 0.8 --num_designs 50 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 180.0 \
#   --screw_translation 0.0 --replicate_count 2 --mode "rotation" --mix_output couple
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-C3-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 120.0 \
  --screw_translation 0.0 --replicate_count 3 --mode "rotation" --mix_output couple
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 300 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-C3-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 120.0 \
  --screw_translation 0.0 --replicate_count 3 --mode "rotation" --mix_output couple
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 200 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-C4-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 90.0 \
  --screw_translation 0.0 --replicate_count 4 --mode "rotation" --mix_output couple
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 400 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-C4-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 90.0 \
  --screw_translation 0.0 --replicate_count 4 --mode "rotation" --mix_output couple
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 250 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C5-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "cosine(t)" --screw_radius 10.0 --screw_angle 72.0 \
  --screw_translation 0.0 --replicate_count 5 --mode "rotation" --mix_output couple

#config=default_ve_minimal_timeless_omap
#path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_ve_minimal_timeless_omap-1024-prod-nsb-default-ve-fixed-1/
config=default_ve_minimal_timeless
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_ve_minimal_timeless-1024-prod-nsb-default-ve-mtl-1/
# screw-monomer stick
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-t12-r0-a30-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "ve(t)" --screw_radius 0.0 --screw_angle 30.0 \
  --screw_translation 12.0 --replicate_count 3 --mode "screw" --mix_output couple
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-t12-r0-a60-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "ve(t)" --screw_radius 0.0 --screw_angle 60.0 \
  --screw_translation 12.0 --replicate_count 3 --mode "screw" --mix_output couple

# screw-monomer offset
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-t12-r10-a30-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "ve(t)" --screw_radius 10.0 --screw_angle 30.0 \
  --screw_translation 12.0 --replicate_count 3 --mode "screw" --mix_output couple
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 150 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-t12-r10-a60-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "ve(t)" --screw_radius 10.0 --screw_angle 60.0 \
  --screw_translation 12.0 --replicate_count 3 --mode "screw" --mix_output couple

# screw-monomer large
python salad/training/eval_sym.py --config $config --params "$path/checkpoint-ema-200000.jax" --num_aa 200 \
  --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-100-t12-r0-a180-1" --num_steps 500 --out_steps "400" \
  --prev_threshold 0.8 --num_designs 50 --timescale_pos "ve(t)" --screw_radius 0.0 --screw_angle 180.0 \
  --screw_translation 16.0 --replicate_count 2 --mode "screw" --mix_output couple


