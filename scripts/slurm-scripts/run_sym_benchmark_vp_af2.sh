#!/bin/bash
#SBATCH -J sym-af2
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 48:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad
AF2PATH=/g/korbel/mjendrusch/repos/alpha-design/
NOVOBENCH=/g/korbel/mjendrusch/repos/novobench/
# vp screw
config=default_vp
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-S3-a120-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-S3-a60-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-S3-a30-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path

# cyclic monomer VE
config=default_ve_minimal_timeless
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C3-r8-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C3-r10-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C3-r12-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C4-r10-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C4-r12-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C5-r12-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C5-r14-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C6-r12-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C6-r14-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C7-r14-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path

config=default_vp
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-C3-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-C3-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-C4-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-C4-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-C5-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path

config=default_ve_minimal_timeless
# screw-monomer stick
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-t12-r0-a30-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-t12-r0-a60-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path

# screw-monomer offset
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-t12-r10-a30-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-t12-r10-a60-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path

# screw-monomer large
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-screw-100-t12-r0-a180-1"
fasta_path=${pdb_path}-sequences/seqs/
out_path=${pdb_path}-af2/
bash $NOVOBENCH/run_novobench.sh --model "af_1" --parameter-path $AF2PATH --prediction-mode "guess" \
    --pdb-path $pdb_path --fasta-path $fasta_path --out-path $out_path


