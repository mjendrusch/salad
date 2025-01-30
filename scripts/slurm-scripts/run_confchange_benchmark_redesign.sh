#!/bin/bash
#SBATCH -J confchange-benchmark-redesign
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
config=default_vp_omap
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_vp_omap-1024-prod-nsb-default-vp-1/
python /g/korbel/mjendrusch/repos/salad/salad/training/eval_split_change_redesign.py --config $config \
  --params "$path/checkpoint-ema-200000.jax" \
  --in_path "/g/korbel/mjendrusch/repos/benchmark_salad/confchange/$config-parent-split-almost-hits/" \
  --num_steps 500 --out_steps "400" --prev_threshold 0.8 --num_designs 1000 --timescale_pos "cosine(t)" \
  --dssp "_HHHHHHHHHHHHHlllhhhhhhhhhhhhhllleeeeeellleeeee_______eeeeellleeeeeelllhhhhhhhhhhhhhlllHHHHHHHHHHHHH_/_HHHHHHHHHHHHHLLLHHHHHHHHHHHHHLLLHHHHHHHHHHHHHHHHHHH_/_HHHHHHHHHHHHHHLLLHHHHHHHHHHHHHLLLHHHHHHHHHHHHH_" \
  --variants 10
# python /g/korbel/mjendrusch/repos/salad/salad/training/eval_split_change.py --config $config \
#   --params "$path/checkpoint-ema-200000.jax" \
#   --out_path "/g/korbel/mjendrusch/repos/benchmark_salad/confchange/$config-parent-split-constrained" \
#   --num_steps 500 --out_steps "400" --prev_threshold 0.8 --num_designs 1000 --timescale_pos "cosine(t)" \
#   --dssp "_HHHHHHHHHHHHHlllHHHHHHHHHHHHHllleeeeeellleeeee_______eeeeellleeeeeelllHHHHHHHHHHHHHlllHHHHHHHHHHHHH_/_HHHHHHHHHHHHHLLLHHHHHHHHHHHHHLLLHHHHHHHHHHHHHHHHHHH_/_HHHHHHHHHHHHHHLLLHHHHHHHHHHHHHLLLHHHHHHHHHHHHH_" \
#   --variants 1
