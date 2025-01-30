#!/bin/bash
#SBATCH -J tmalign-all-to-all
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p htc-el8
#SBATCH --mem 32G
#SBATCH -t 04:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

NAME=$1
IN_PATH=$2
OUT_PATH=$3
# find $IN_PATH -maxdepth 1 -name "*.pdb" | sort | xargs -i basename {} > $OUT_PATH/chain_list_${NAME}_all
python scripts/success_chainlist.py $IN_PATH $OUT_PATH/chain_list_${NAME}
TMalign -dir $IN_PATH/ -fast -outfmt 2 $OUT_PATH/chain_list_$NAME > $OUT_PATH/tmalign_$NAME