#!/bin/bash
#SBATCH -J salad-tmalign-single
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -p htc-el8
#SBATCH --mem 32G
#SBATCH -t 12:00:00
#SBATCH --mail-user=your@email.io
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN,END

source $HOME/.bashrc
cd /g/korbel/mjendrusch/repos/salad/
conda activate salad

NOVOBENCH=/g/korbel/mjendrusch/repos/novobench/
motif_path=/g/korbel/mjendrusch/repos/genie2/data/design25/
mode=$1
# for motif in `ls $motif_path`; do
#     echo $motif
#     motif_name=$(basename motif)
#     bash scripts/tmalign.sh tmalign_alltoall_${mode} \
#         "/g/korbel/mjendrusch/repos/benchmark_salad/motif/cond/multimotif_vp-$motif-esm/success_${mode}/" \
#         "/g/korbel/mjendrusch/repos/benchmark_salad/motif/cond/multimotif_vp-$motif-esm/" &
#     bash scripts/tmalign.sh tmalign_alltoall_${mode} \
#         "/g/korbel/mjendrusch/repos/benchmark_salad/motif/nocond/default_vp-$motif-esm/success_${mode}/" \
#         "/g/korbel/mjendrusch/repos/benchmark_salad/motif/nocond/default_vp-$motif-esm/" &
# done

motif_path=/g/korbel/mjendrusch/repos/genie2/data/multimotifs/
for motif in `ls $motif_path`; do
    echo $motif
    motif_name=$(basename motif)
    bash scripts/tmalign.sh tmalign_alltoall_${mode} \
        "/g/korbel/mjendrusch/repos/benchmark_salad/motif/cond/multimotif_vp-$motif-esm/success_${mode}/" \
        "/g/korbel/mjendrusch/repos/benchmark_salad/motif/cond/multimotif_vp-$motif-esm/" &
    bash scripts/tmalign.sh tmalign_alltoall_${mode} \
        "/g/korbel/mjendrusch/repos/benchmark_salad/motif/nocond/default_vp-$motif-esm/success_${mode}/" \
        "/g/korbel/mjendrusch/repos/benchmark_salad/motif/nocond/default_vp-$motif-esm/" &
done
wait