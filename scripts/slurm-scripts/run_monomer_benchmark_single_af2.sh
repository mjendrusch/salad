#!/bin/bash
#SBATCH -J salad-esm-single
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

AF2PATH=/g/korbel/mjendrusch/repos/alpha-design/
NOVOBENCH=/g/korbel/mjendrusch/repos/novobench/
config=$1
num_aa=300
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-500s/" \
    --fasta-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-500s-sequences/seqs/" \
    --out-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-500s-af2/"
for num_aa in 400; do
  bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-100s/" \
    --fasta-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-100s-sequences/seqs/" \
    --out-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-100s-af2/"
  bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-200s/" \
    --fasta-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-200s-sequences/seqs/" \
    --out-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-200s-af2/"
  bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-500s/" \
    --fasta-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-500s-sequences/seqs/" \
    --out-path "/g/korbel/mjendrusch/repos/benchmark_salad/monomers/$config-${num_aa}-500s-af2/"
done
