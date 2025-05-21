"""Basic usage example of the inference API."""

from copy import deepcopy
import os
import time

import jax
import haiku as hk

import salad.inference as si

from flexloop.utils import parse_options

# here, we implement a denoising step
def model_step(config):
    # get the configuration and set eval to true
    # this turns off any model components that
    # are only used during training.
    config = deepcopy(config)
    config.eval = True
    # salad is built on top of haiku, which means
    # that any functions using salad modules need
    # to be hk.transform-ed before use.
    @hk.transform
    def step(data, prev):
        # instantiate a noise generator
        noise = si.StructureDiffusionNoise(config)
        # and a denoising model
        predict = si.StructureDiffusionPredict(config)
        # we can edit the structure before noise
        # is applied here:
        ...
        # apply noise
        data.update(noise(data))
        # and edit the noised structure here:
        ...
        # run model
        out, prev = predict(data, prev)
        # or the output of the model here:
        ...
        return out, prev
    # return the pure apply function generated
    # by haiku from our step
    return step.apply

opt = parse_options(
    "Example script using the salad.inference API",
    params="checkpoint.jax",
    out_path="output/",
    num_aa="100",
    num_designs=10,
)

os.makedirs(opt.out_path, exist_ok=True)
key = si.KeyGen(42)
config, params = si.make_salad_model("default_vp", opt.params)

# initialize data and prev from the num_aa specification
num_aa, resi, chain, is_cyclic, cyclic_mask = si.parse_num_aa(opt.num_aa)
data, init_prev = si.data.from_config(
    config,
    num_aa=num_aa,
    residue_index=resi,
    chain_index=chain,
    cyclic_mask=cyclic_mask)

step = model_step(config)

# build a sampler object for sampling
sampler = si.Sampler(step, out_steps=400)
# run a loop with num_design steps
print("Starting design...")
for idx in range(opt.num_designs):
    # generate a structure in each step
    start = time.time()
    design = sampler(params, key(), data, init_prev)
    print(f"Design {idx} in {time.time() - start:.2f} s")
    # and write it to PDB
    with open(f"{opt.out_path}/result_{idx}.pdb", "wt") as f:
        f.write(si.data.to_pdb(design))
