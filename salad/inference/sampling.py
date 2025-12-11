"""Utilities for writing salad scripts: implementation of the sampling loop."""

from typing import Callable
import numpy as np

import jax
import jax.numpy as jnp

from salad.inference.utils import parse_timescale

class Sampler:
    def __init__(self, step, num_steps=500, out_steps=400,
                 start_steps=0, prev_threshold=0.8,
                 timescale="cosine(t)",
                 timescale_seq=None):
        self.step = jax.jit(step)
        self.num_steps = num_steps
        self.out_steps = out_steps
        self.start_steps = start_steps
        self.prev_threshold = prev_threshold
        self.timescale = Sampler._get_timescale(timescale)
        self.timescale_seq = Sampler._get_timescale(timescale_seq) if timescale_seq is not None else None

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
            timescale=self.timescale,
            timescale_seq=self.timescale_seq)

    def generate(self, params, data, prev):
        while True:
            key, subkey = jax.random.split(key, 2)
            yield self(params, subkey, data, prev)

def sample(step, params, key, data, prev,
           num_steps=500, out_steps=400, start_steps=0,
           prev_threshold=0.8, timescale="cosine(t)",
           timescale_seq=None):
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
        scaled_t_seq = 1.0
        if timescale_seq is not None:
            scaled_t_seq = timescale_seq(raw_t)
        data["t_pos"] = scaled_t * jnp.ones_like(data["t_pos"])
        data["t_seq"] = scaled_t_seq * jnp.ones_like(data["t_seq"])
        if raw_t < prev_threshold:
            prev = init_prev
        update, prev = step(params, subkey, data, prev)
        pos = update["pos"]
        data["pos"] = pos
        if ("latent" in data) and ("latent" in update):
            data["latent"] = update["latent"]
        data["seq"] = jnp.argmax(update["aa"], axis=-1)
        data["atom_pos"] = update["atom_pos"]
        data["aatype"] = update["aatype"]
        if idx in out_steps:
            data["result"] = update
            data["aa_logits"] = update["aa"]
            results.append({key: value for key, value in data.items()})
        if idx == out_steps[-1]:
            break
    if return_list:
        return results
    return results[0]
