#!/bin/bash
#SBATCH -J autosym-af2
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 16:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad
AF2PATH=/g/korbel/mjendrusch/repos/alpha-design/
NOVOBENCH=/g/korbel/mjendrusch/repos/novobench/
# autosym vp
config=default_vp
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_vp-1024-prod-nsb-default-vp-1/
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-2-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-3-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-4-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
# complex
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-autosym-cpx-2-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-2-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-3-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-4-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path

# cyclic monomer VE
config=default_ve_minimal_timeless
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_ve_minimal_timeless-1024-prod-nsb-default-ve-fixed-1/
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-2-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-3-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-4-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
# complex
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-autosym-cpx-2-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-2-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-3-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-4-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
