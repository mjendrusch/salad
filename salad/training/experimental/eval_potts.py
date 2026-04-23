import os
import pickle

from copy import deepcopy
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from torch.utils.tensorboard import SummaryWriter

from flexloop.simple_loop import (
    training, log, load_loop_state, update_step, valid_step, rebatch_call, State)
from salad.data.convert import parse_structure, read_components
from salad.data.allpdb import slice_dict, AA_ORDER, ATOM_TYPE_ORDER, numerical_chain_index
from salad.modules.experimental.old_sequence_design import SequenceModel
from salad.modules.config import old_sequence_design as config_choices
from flexloop.utils import parse_options
from flexloop.loop import cast_float

def model_step(config):
    model = SequenceModel
    config = deepcopy(config)
    config.eval = True
    def step(data):
        data = jax.tree_util.tree_map(lambda x: jnp.array(x), data)
        data = cast_float(data, dtype=jnp.float32)
        loss, out = model(config)(data)
        return cast_float(loss, jnp.float32), cast_float(out["results"]["potts"], jnp.float32)
    return step

# TODO: refactor, implementing data transformations
def parse_pdb(components, path):
    raw_data = parse_structure(components, path)
    aa_order = np.array(AA_ORDER)
    atom_type_order = np.array(ATOM_TYPE_ORDER)
    aa_index = raw_data["residue_type"] == "AA"
    aa_data = slice_dict(raw_data, aa_index)
    aa_gt = np.argmax(
        aa_data["residue_name"][:, None] == aa_order[None, :], axis=-1)
    aa_gt = np.where(
        (aa_data["residue_name"][:, None] != aa_order[None, :]).all(axis=-1),
        20,
        aa_gt
    )
    residue_index = aa_data["residue_index"]
    chain_index = numerical_chain_index(aa_data["chain_index"])
    all_atom_positions = aa_data["position"]
    all_atom_mask = aa_data["atom_mask"]

    # get small molecule / non-protein data
    not_aa_data = slice_dict(raw_data, ~aa_index)
    atom_mask = not_aa_data["atom_mask"]
    atom_positions = not_aa_data["position"][atom_mask]
    atom_types = not_aa_data["atom_type"][atom_mask]
    assignment = atom_types[:, None] == atom_type_order[None, :]
    assignment = np.where(assignment.any(axis=-1), np.argmax(assignment, axis=-1), len(ATOM_TYPE_ORDER) + 1)
    smol_positions = atom_positions
    smol_types = assignment
    
    # get smol neighbours
    ca = all_atom_positions[:, 1]
    distance = np.linalg.norm(ca[:, None] - smol_positions[None, :], axis=-1)
    neighbours = np.argsort(distance, axis=1)[:, :16]
    if neighbours.shape[1] < 16:
        diff = 16 - neighbours.shape[1]
        neighbours = np.concatenate((
            neighbours,
            -np.ones((neighbours.shape[0], diff),
                    dtype=np.int32)), axis=1)
    if smol_positions.shape[0] == 0:
        smol_positions = np.zeros(list(neighbours.shape) + [3], dtype=np.float32)
        smol_types = np.zeros(neighbours.shape, dtype=np.int32)
        smol_mask = np.zeros(neighbours.shape, dtype=np.bool_)
    else:
        smol_positions = smol_positions[neighbours]
        smol_types = smol_types[neighbours]
        smol_mask = neighbours != -1
        smol_positions = np.where(smol_mask[..., None], smol_positions, 0)
        smol_types = np.where(smol_mask, smol_types, 6)

    return dict(
        aa_gt=aa_gt,
        residue_index=residue_index,
        chain_index=chain_index,
        batch_index=np.zeros_like(chain_index),
        all_atom_positions=all_atom_positions,
        all_atom_mask=all_atom_mask,
        smol_positions=smol_positions,
        smol_types=smol_types,
        smol_mask=smol_mask,
        mask=np.ones_like(chain_index, dtype=np.bool_)
    )

if __name__ == "__main__":

    opt = parse_options(
        "train a protein diffusion model on PDB.",
        config="default",
        ccd_path="/g/korbel/mjendrusch/data/allpdb/components.cif.gz",
        pdb="test.pdb",
        out_path="potts_test/",
        param_path="params.jax",
        ensemble=1,
        jax_seed=42
    )
    os.makedirs(opt.out_path, exist_ok=True)

    print("Attempting to load file...")
    components = read_components(opt.ccd_path)
    item = parse_pdb(components, opt.pdb)

    config = getattr(config_choices, opt.config)
    key = jax.random.PRNGKey(opt.jax_seed)
    _, valid = hk.transform(model_step(config))
    valid = jax.jit(valid)

    with open(opt.param_path, "rb") as f:
        params = pickle.load(f)

    print("Initializing model parameters...")
    base_pos = item["all_atom_positions"]
    if opt.ensemble < 2:
        loss, out = valid(params, key, item)
    else:
        h_i = 0.0
        J_ij = 0.0
        for _ in range(opt.ensemble):
            item["all_atom_positions"] = base_pos + 0.3 * np.random.randn(*base_pos.shape)
            loss, out = valid(params, key, item)
            h_i += out["pssm_term"] / opt.ensemble
            index = np.arange(h_i.shape[0])[:, None]
            J_u = jnp.zeros((h_i.shape[0], h_i.shape[0], 20, 20), dtype=jnp.float32)
            J_u = J_u.at[index, out["neighbours"]].set(out["contact_term"])
            J_ij += J_u / opt.ensemble
        J_ij = J_ij[index, out["neighbours"]]
        out["pssm_term"] = h_i
        out["contact_term"] = J_ij
    basename = ".".join(os.path.basename(opt.pdb).split(".")[:-1])
    np.savez_compressed(f"{opt.out_path}/{basename}.npz", **{k: np.array(v) for k, v in item.items()}, **{k: np.array(v) for k, v in out.items()})
    print("Model parameters initialized.")
