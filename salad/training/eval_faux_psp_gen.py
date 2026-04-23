import os
# FIXME: why is jax trying to use a triton kernel here?
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
import time

from copy import deepcopy
import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk
import optax

# from torch.utils.tensorboard import SummaryWriter

# from flexloop.simple_loop import (
#     training, log, load_loop_state, update_step, rebatch_call, State)
from salad.modules.experimental.faux_psp import StructureEncoding, StructureHal
from salad.modules.config import faux_psp as config_choices
from flexloop.utils import parse_options
# from flexloop.loop import cast_float

def model_step(config, softmax=True, repeat=None):
    module = StructureHal
    config = deepcopy(config)
    config.eval = True
    config.encoder.eval = True
    config.decoder.eval = True
    def step(data):
        if repeat is not None and repeat > 1:
            data = repeat_data(data, repeat=repeat)
        if softmax:
            data = jax.nn.softmax(data, axis=-1)
        loss, out = module(config)(data)
        return loss, out
    return step

def repeat_data(data, repeat=2):
    return jnp.concatenate(repeat * [data], axis=0)

def repeat_op(data, repeat=2):
    base = jnp.zeros((repeat, data.shape[0], data.shape[1]), dtype=data.dtype)
    return (base + data).reshape(-1, data.shape[1])

def update_step(the_step, optimizer):
    def inner(params, opt_state, key, latent):
        (value, out), grad = jax.value_and_grad(
            the_step, argnums=(2,), has_aux=True)(params, key, latent)
        grad = grad[0]
        grad /= jnp.maximum(jnp.linalg.norm(grad, axis=(-1, -2), keepdims=True), 1e-3)
        updates, opt_state = optimizer.update(grad, opt_state, params=latent)
        latent = optax.apply_updates(latent, updates)
        # latent -= lr * grad
        #jax.tree_util.tree_map(lambda v, g: v - lr * g, latent, grad)
        return value, latent, opt_state, out
    return inner

def repeat_dict(x, count):
    return {
        key: np.concatenate(count * [value], axis=0)
        for key, value in x.items()
    }

AA_CODE = "ARNDCQEGHILKMFPSTWYVX"
def decode_sequence(x: np.ndarray) -> str:
    x = np.array(x)
    return "".join([AA_CODE[c] for c in x])

if __name__ == "__main__":
    import numpy as np
    from salad.data.allpdb import CroppedPDBStream
    from flexloop.data import BatchStream
    from salad.aflib.common.protein import from_pdb_string, to_pdb, Protein
    from salad.aflib.model.all_atom_multimer import atom37_to_atom14, atom14_to_atom37, get_atom37_mask
    from salad.aflib.model.geometry import Vec3Array
    import pickle

    # TODO: this requires changes to the model for use with
    # gradient descent (remove encoder, use just decoder, re-enable gradients)

    opt = parse_options(
        "eval a fpsp model.",
        out_path="out/",
        num_aa=100,
        config="default",
        num_recycle=2,
        jax_seed=57,
        params="checkpoint.jax",
        softmax="True",
        latent_scale=5.0,
        repeat=1
    )
    os.makedirs(opt.out_path, exist_ok=True)
    config = getattr(config_choices, opt.config)
    config.decoder.num_recycle = opt.num_recycle
    config.decoder.gen = True
    config.gen = True
    key = jax.random.PRNGKey(opt.jax_seed)
    _, step = hk.transform(model_step(config, softmax=opt.softmax == "True", repeat=opt.repeat))
    optimizer = optax.chain(
        # scale gradients using Adam
        optax.scale_by_adam(0.9, 0.99, eps=1e-9),
        # scale resulting learning rate by a cosine schedule
        optax.scale(-0.1))
    ustep = (update_step(jax.jit(step), optimizer))
    with open(opt.params, "rb") as f:
        params = pickle.load(f)


    AA_CODE = "ARNDCQEGHILKMFPSTWYVX"
    for idx in range(10):
        latent = jnp.array(
            opt.latent_scale * np.random.randn(opt.num_aa, config.encoder.latent_size), dtype=jnp.float32)
        if config.encoder.kind == "AAEncoder":
            if opt.softmax == "True":
                latent = np.random.randn(opt.num_aa, 20)
                latent[:, AA_CODE.index("C")] = -1e9
            else:
                latent = np.random.rand(opt.num_aa, 20)
                latent /= jnp.maximum(latent.sum(axis=1, keepdims=True), 1e-3)
        # set up optimizer
        opt_state = optimizer.init(latent)
        total = 1e6
        ids = 0
        while total > 2.0 or ids < 50:
            total, latent, opt_state, out = ustep(params, opt_state, key, latent)
            # if opt.softmax == "True":
            #     latent = latent.at[:, AA_CODE.index("C")].set(-1e9)
            # else:
            #     latent = latent.at[:, AA_CODE.index("C")].set(0)
            # latent = jnp.clip(latent, -6, 6)
            print("step", ids, "total", total)
            if config.encoder.kind == "AAEncoder":
                print(decode_sequence(np.argmax(latent, axis=-1)))
            ids += 1
        final_struc = out["trajectory"][-1]
        aatype = np.argmax(out["aa"], axis=-1)
        atom37 = atom14_to_atom37(final_struc, aatype)
        atom37_mask = get_atom37_mask(aatype)
        residue_index = np.arange(opt.num_aa * opt.repeat, dtype=np.int32)
        mean_pae = np.ones_like(atom37_mask, dtype=jnp.float32)
        if "pae" in out:
            pae = (np.exp(out["pae"]) * np.linspace(0.0, 40.0, 64)).sum(axis=-1)
            mu_pae = pae.mean(axis=1)
            mean_pae *= mu_pae[:, None]
        protein = Protein(np.array(atom37), np.array(aatype),
                        np.array(atom37_mask), residue_index,
                        np.zeros_like(residue_index), mean_pae)
        pdb_path = f"{opt.out_path}/result_{idx}_{ids}.pdb"
        pdb_string = to_pdb(protein)
        with open(pdb_path, "wt") as f:
            f.write(pdb_string)
