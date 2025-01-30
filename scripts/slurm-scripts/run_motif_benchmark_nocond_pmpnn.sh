#!/bin/bash
#SBATCH -J confchange-benchmark-pmpnn
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 20:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad
AF2PATH=/g/korbel/mjendrusch/repos/alpha-design/
NOVOBENCH=/g/korbel/mjendrusch/repos/novobench/

basepath=/g/korbel/mjendrusch/repos/benchmark_salad/motif/cond/multimotif_vp
# single motif vp
# for target_name in `ls /g/korbel/mjendrusch/repos/genie2/data/design25/`; do
#   echo $target_name
#   target_path=/g/korbel/mjendrusch/repos/genie2/data/design25/$target_name
#   pdb_path=${basepath}-${target_name}
#   af2_path=${basepath}-${target_name}-af2
#   esm_path=${basepath}-${target_name}-esm
#   fasta_path=${basepath}-${target_name}-sequences/seqs/
#   sbr=$(sbatch scripts/run_pmpnn_fixed.sh $pdb_path $target_path ${basepath}-${target_name}-sequences)
#   if [[ "$sbr" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
#       JID="${BASH_REMATCH[1]}"
#   else
#       echo "sbatch failed"
#   fi
#   DEP="afterok:$JID"
#   # sbatch -n 32 -p gpu-el8 -C gpu=L40s --gres=gpu:L40s:1 -t 24:00:00 --dependency=$DEP $NOVOBENCH/run_novobench.sh \
#   #   --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
#   #   --pdb-path $pdb_path --fasta-path $fasta_path --out-path $af2_path/
#   sbatch -n 32 -p gpu-el8 -C gpu=L40s --gres=gpu:L40s:1 -t 32:00:00 --dependency=$DEP $NOVOBENCH/run_novobench.sh \
#     --pdb-path $pdb_path --fasta-path $fasta_path --out-path $esm_path/
# done

# multi motif vp
for target_name in `ls /g/korbel/mjendrusch/repos/genie2/data/multimotifs/`; do
  echo $target_name
  target_path=/g/korbel/mjendrusch/repos/genie2/data/multimotifs/$target_name
  pdb_path=${basepath}-${target_name}
  fasta_path=${basepath}-${target_name}-sequences/seqs/
  af2_path=${basepath}-${target_name}-af2
  esm_path=${basepath}-${target_name}-esm
  sbr=$(sbatch scripts/run_pmpnn_fixed.sh $pdb_path $target_path ${basepath}-${target_name}-sequences)
  if [[ "$sbr" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
     JID="${BASH_REMATCH[1]}"
  else
     echo "sbatch failed"
  fi
  DEP=afterok:${JID}
  # DEP="afterok:14875964,afterok:14875966"
  # sbatch -n 32 -p gpu-el8 -C gpu=L40s --gres=gpu:L40s:1 -t 24:00:00 --dependency=$DEP $NOVOBENCH/run_novobench.sh \
  #   --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
  #   --pdb-path $pdb_path --fasta-path $fasta_path --out-path $af2_path/
  sbatch -n 32 -p gpu-el8 -C gpu=L40s --gres=gpu:L40s:1 -t 32:00:00 --dependency=$DEP $NOVOBENCH/run_novobench.sh \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $esm_path/
done
