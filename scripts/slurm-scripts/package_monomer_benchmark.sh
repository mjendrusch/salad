#!/bin/bash
#SBATCH -J salad-large-esm
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

BASE=/g/korbel/mjendrusch/repos/benchmark_salad/

# step benchmark
# for config in default_vp default_vp_timeless default_vp_minimal_timeless default_vp_scaled default_vp_scaled_timeless default_vp_scaled_minimal_timeless default_ve_timeless default_ve_minimal_timeless ve_large_80 ve_large_100; do
#   for num_aa in 50 100 200 300 400; do
#     for steps in 100s 200s 500s random; do
#       python scripts/package_denovo.py $BASE/monomers/${config}-${num_aa}-${steps} $BASE/data_package/monomers/${config}-${num_aa}-${steps} esm
#     done
#     cp $BASE/monomers/stats-${config}-${num_aa}-500s.csv $BASE/data_package/monomers/${config}-${num_aa}-${steps}-esm/stats-${config}-${num_aa}-500s.csv
#     cp $BASE/monomers/tmalign_${config}_${num_aa}_500s $BASE/data_package/monomers/${config}-${num_aa}-${steps}-esm/tmalign_${config}_${num_aa}_500s
#     cp $BASE/monomers/tmalign_${config}_${num_aa}_random $BASE/data_package/monomers/${config}-${num_aa}-${steps}-esm/tmalign_${config}_${num_aa}_random
#   done
# done

# large
# steps=500s
# for config in default_vp default_vp_scaled ve_large_80 ve_large_100; do
#   for num_aa in 500 600 800 1000; do
#     python scripts/package_denovo.py $BASE/monomers/${config}-${num_aa}-${steps} $BASE/data_package/monomers/${config}-${num_aa}-${steps} esm
#     cp $BASE/monomers/stats-${config}-${num_aa}-500s.csv $BASE/data_package/monomers/${config}-${num_aa}-${steps}-esm/stats-${config}-${num_aa}-500s.csv
#   done
# done

# # large-random
# steps=random
# for config in ve_large_80; do
#   for num_aa in 500 600 800 1000; do
#     python scripts/package_denovo.py $BASE/monomers/${config}-${num_aa}-${steps} $BASE/data_package/monomers/${config}-${num_aa}-${steps} esm
#     cp $BASE/monomers/stats-${config}-${num_aa}-500s.csv $BASE/data_package/monomers/${config}-${num_aa}-${steps}-esm/stats-${config}-${num_aa}-500s.csv
#   done
# done

# domain
# steps=500s
# for config in ve_domain_80 ve_domain_100; do
#   for num_aa in 400 500 600 800 1000; do
#     python scripts/package_denovo.py $BASE/monomers/${config}-${num_aa}-${steps} $BASE/data_package/monomers/${config}-${num_aa}-${steps} esm
#     cp $BASE/monomers/stats-${config}-${num_aa}-500s.csv $BASE/data_package/monomers/${config}-${num_aa}-${steps}-esm/stats-${config}-${num_aa}-500s.csv
#   done
# done

# letter shapes
# for letter in S A L D; do
#   python scripts/package_denovo.py $BASE/shape/ve-seg-${letter}-1 $BASE/data_package/shape/ve-seg-${letter}-1 esm
# done

# sym
# for spec in default_ve_minimal_timeless-screw-100-t12-r0-a180-1 default_ve_minimal_timeless-screw-t12-r0-a30-1 default_ve_minimal_timeless-screw-t12-r0-a60-1 default_ve_minimal_timeless-screw-t12-r10-a30-1 default_ve_minimal_timeless-screw-t12-r10-a60-1 default_vp-100-C3-1 default_vp-100-C4-1 default_vp-50-C3-1 default_vp-50-C4-1 default_ve_minimal_timeless-C3-r10-1 default_ve_minimal_timeless-C3-r12-1 default_ve_minimal_timeless-C3-r8-1 default_ve_minimal_timeless-C4-r10-1 default_ve_minimal_timeless-C4-r12-1 default_ve_minimal_timeless-C5-r12-1 default_ve_minimal_timeless-C5-r14-1 default_ve_minimal_timeless-C6-r12-1 default_ve_minimal_timeless-C6-r14-1 default_ve_minimal_timeless-C7-r14-1; do
#   python scripts/package_denovo.py $BASE/sym/${spec} $BASE/data_package/sym/${spec} esm
# done

# motif
# for condition in cond; do
#   for name in 1bcf 1prw_four 1prw 1prw_two 1qjg 1ycr 2b5i 2kl8 3bik+3bp5 3ixt 3ntn 4jhw+5wn9 4jhw 4zyp 5ius 5tpn 5trv_long 5trv_med 5trv_short 5wn9 5yui 6e6r_long 6e6r_med 6e6r_short 6exz_long 6exz_med 6exz_short 7mrx_128 7mrx_60 7mrx_85; do
#     python scripts/package_motif.py $BASE/motif/${condition}/multimotif_vp-${name}.pdb $BASE/data_package/motif/${condition}/multimotif_vp-${name}.pdb
#   done
# done
# for condition in nocond; do
#   for name in 1bcf 1prw_four 1prw 1prw_two 1qjg 1ycr 2b5i 2kl8 3bik+3bp5 3ixt 3ntn 4jhw+5wn9 4jhw 4zyp 5ius 5tpn 5trv_long 5trv_med 5trv_short 5wn9 5yui 6e6r_long 6e6r_med 6e6r_short 6exz_long 6exz_med 6exz_short 7mrx_128 7mrx_60 7mrx_85; do
#     python scripts/package_motif.py $BASE/motif/${condition}/default_vp-${name}.pdb $BASE/data_package/motif/${condition}/default_vp-${name}.pdb
#   done
# done

# confchange
# for spec in default_vp-parent-split default_vp-parent-split-constrained; do
#   python scripts/package_confchange.py $BASE/confchange/${spec} $BASE/data_package/confchange/${spec}
# done

# synthetic
for config in pdb256 synthete256; do
  for num_aa in 50 100 200 300 400; do
    for steps in 100s 200s 500s; do
      python scripts/package_denovo.py $BASE/synthete/${config}-${num_aa}-${steps} $BASE/data_package/synthete/${config}-${num_aa}-${steps} esm
    done
  done
done
