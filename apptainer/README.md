# Apptainer images for salad
### Setup
To build an apptainer image for salad, run:
```
apptainer build salad.sif apptainer/salad.def
```
This will create an image `salad.sif` and take around 10-20 minutes to complete.

### Running salad
To run salad for protein design, first download the parameters as detailed in the salad install instructions.
Then, run salad using the apptainer image, e.g.:
```
apptainer run --nv salad.sif -m salad.training.eval_noise_schedule_benchmark \
    --config default_vp \
    --params params/default_vp-200k.jax \
    --out_path output/my_first_protein/ \
    --num_aa 100 \
    --num_designs 10
```
To run another salad script, specify its module name and parameters instead.
