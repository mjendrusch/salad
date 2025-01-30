#!/bin/bash
#SBATCH -J autosym-pmpnn
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p gpu-el8
#SBATCH -C gpu=L40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --mem-per-gpu 32000
#SBATCH -t 08:00:00
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
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-2-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_tied.sh $pdb_path $fasta_path 50 2
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-3-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_tied.sh $pdb_path $fasta_path 50 3
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-4-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_tied.sh $pdb_path $fasta_path 50 4
# complex
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-autosym-cpx-2-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_homomer.sh $pdb_path $fasta_path 100 2
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-2-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_homomer.sh $pdb_path $fasta_path 50 2
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-3-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_homomer.sh $pdb_path $fasta_path 50 3
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-4-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_homomer.sh $pdb_path $fasta_path 50 4

# cyclic monomer VE
config=default_ve_minimal_timeless
path=/g/korbel/mjendrusch/runs/debug-salad-new/salad/nsb-default_ve_minimal_timeless-1024-prod-nsb-default-ve-fixed-1/
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-2-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_tied.sh $pdb_path $fasta_path 50 2
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-3-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_tied.sh $pdb_path $fasta_path 50 3
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-4-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_tied.sh $pdb_path $fasta_path 50 4
# complex
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-100-autosym-cpx-2-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_homomer.sh $pdb_path $fasta_path 100 2
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-2-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_homomer.sh $pdb_path $fasta_path 50 2
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-3-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_homomer.sh $pdb_path $fasta_path 50 3
pdb_path="/g/korbel/mjendrusch/repos/benchmark_salad/sym/$config-50-autosym-cpx-4-1"
fasta_path=${pdb_path}-sequences/
bash scripts/run_pmpnn_homomer.sh $pdb_path $fasta_path 50 4
