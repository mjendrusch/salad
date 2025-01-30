#!/bin/bash
#SBATCH -J salad-pmpnn-fixed
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p gpu-el8
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 24:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
module load CUDA/12.0.0

TARGET=$1
path_to_motif=$2
OUT=$3
PROTEINMPNN=/g/korbel/mjendrusch/repos/rf_diffusion_env/ProteinMPNN/
mpnnseed=37
folder_with_pdbs=$TARGET
output_dir=$OUT
mkdir -p $output_dir

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_fixed_positions=$output_dir"/fixed_positions.jsonl"
chains_to_design="A"

conda activate salad
python scripts/motif_pmpnn_json.py \
        $folder_with_pdbs $path_to_motif $output_dir \
        $path_for_fixed_positions $path_for_parsed_chains 2
conda deactivate

conda activate SE3nv
python $PROTEINMPNN/protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --fixed_positions_jsonl $path_for_fixed_positions \
        --out_folder $output_dir \
        --num_seq_per_target 10 \
        --sampling_temp "0.1" \
        --model_name "v_48_030" \
        --seed $mpnnseed \
        --batch_size 1
