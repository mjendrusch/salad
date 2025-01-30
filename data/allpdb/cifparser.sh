conda activate salad

for N in `seq 1 8`; do
  python cif2npz.py assembly &
done
wait
