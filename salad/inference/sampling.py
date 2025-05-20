from typing import Callable
import numpy as np

import jax
import jax.numpy as jnp

from salad.aflib.common.protein import Protein, to_pdb
from salad.aflib.model.all_atom_multimer import atom14_to_atom37, get_atom37_mask

from salad.inference.utils import parse_timescale

class Sampler:
    def __init__(self, step, num_steps=500, out_steps=400,
                 start_steps=0, prev_threshold=0.8, timescale="cosine(t)"):
        self.step = jax.jit(step)
        self.num_steps = num_steps
        self.out_steps = out_steps
        self.start_steps = start_steps
        self.prev_threshold = prev_threshold
        self.timescale = Sampler._get_timescale(timescale)

    @staticmethod
    def _get_timescale(timescale: str | Callable[[float], float]) -> Callable[[float], float]:
        if isinstance(timescale, str):
            timescale = parse_timescale(timescale)
        return timescale
    
    def __call__(self, params, key, data, prev):
        return sample(
            self.step, params, key, data, prev,
            num_steps=self.num_steps,
            out_steps=self.out_steps,
            prev_threshold=self.prev_threshold,
            timescale=self.timescale)

    def generate(self, params, data, prev):
        while True:
            key, subkey = jax.random.split(key, 2)
            yield self(params, subkey, data, prev)

def sample(step, params, key, data, prev,
           num_steps=500, out_steps=400, start_steps=0,
           prev_threshold=0.8, timescale="cosine(t)"):
    if isinstance(timescale, str):
        timescale = parse_timescale(timescale)
    return_list = True
    if not isinstance(out_steps, (list, tuple)):
        return_list = False
        out_steps = [out_steps]
    data = {key: value for key, value in data.items()}
    results = []
    init_prev = prev
    prev = init_prev
    for idx in range(start_steps, num_steps):
        key, subkey = jax.random.split(key)
        raw_t = 1 - idx / num_steps
        scaled_t = timescale(raw_t)
        data["t_pos"] = scaled_t * jnp.ones_like(data["t_pos"])
        data["t_seq"] = jnp.ones_like(data["t_seq"])
        if raw_t < prev_threshold:
            prev = init_prev
        update, prev = step(params, subkey, data, prev)
        pos = update["pos"]
        data["pos"] = pos
        data["seq"] = jnp.argmax(update["aa"], axis=-1)
        data["atom_pos"] = update["atom_pos"]
        data["aatype"] = update["aatype"]
        if idx in out_steps:
            data["result"] = update
            results.append({key: value for key, value in data.items()})
        if idx == out_steps[-1]:
            break
    if return_list:
        return results
    return results[0]

def init_data(config, num_aa=None,
              residue_index=None,
              chain_index=None,
              cyclic_mask=None):
    c = config
    if cyclic_mask is not None:
        c.cyclic = True
    if num_aa is not None:
        if chain_index is None:
            chain_index = jnp.zeros((num_aa,), dtype=jnp.int32)
        if residue_index is None:
            residue_index = jnp.arange(num_aa, dtype=jnp.int32)
    else:
        if chain_index is not None:
            num_aa = chain_index.shape[0]
        if residue_index is not None:
            num_aa = residue_index.shape[0]
    init_pos = jnp.zeros((num_aa, 5 + c.augment_size, 3), dtype=jnp.float32)
    init_aa_gt = jnp.full((num_aa,), 20, dtype=jnp.int32)
    init_local = jnp.zeros((num_aa, c.local_size), dtype=jnp.float32)
    data = dict(
        pos=init_pos,
        aa_gt=init_aa_gt,
        seq=init_aa_gt,
        residue_index=residue_index,
        chain_index=chain_index,
        batch_index=jnp.zeros_like(chain_index),
        mask=jnp.ones_like(chain_index, dtype=jnp.bool_),
        cyclic_mask=cyclic_mask,
        t_pos=jnp.ones((num_aa,), dtype=jnp.float32),
        t_seq=jnp.ones((num_aa,), dtype=jnp.float32)
    )
    prev = dict(
        pos=jnp.zeros_like(init_pos),
        local=init_local
    )
    return data, prev

def update_data(data, **kwargs):
    data = {k: v for k, v in data.items()}
    for name, item in kwargs.items():
        data[name] = item
    return data

def data_to_protein(data):
    atom37 = atom14_to_atom37(data["atom_pos"], data["aatype"])
    atom37_mask = get_atom37_mask(data["aatype"])
    residue_index = np.array(data["residue_index"])
    if "result" in data:
        maxprob = jax.nn.softmax(data["result"]["aa"], axis=-1).max(axis=-1)
    else:
        maxprob = np.zeros((residue_index.shape[0],), dtype=np.float32)
    protein = Protein(
        np.array(atom37),
        np.array(data["aatype"]),
        np.array(atom37_mask),
        residue_index,
        np.array(data["chain_index"]),
        maxprob[:, None] * np.array(
            np.ones_like(atom37_mask, dtype=jnp.float32)))
    return protein
