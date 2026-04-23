import time

from copy import deepcopy

import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from torch.utils.tensorboard import SummaryWriter

from flexloop.simple_loop import (
    training, log, load_loop_state, update_step, rebatch_call, State)
from salad.modules.experimental.salad_simple import SimpleDiffusion
from salad.modules.config import salad_simple as config_choices
from salad.aflib.common.protein import Protein, to_pdb
from salad.aflib.model.all_atom_multimer import atom14_to_atom37, get_atom37_mask
from flexloop.utils import parse_options
from flexloop.loop import cast_float

def model_step(config, rebatch=1, is_training=True):
    module = SimpleDiffusion
    if not is_training:
        config = deepcopy(config)
        config.eval = True
    def step(data):
        data = jax.tree_util.tree_map(lambda x: jnp.array(x), data)
        data = cast_float(data, dtype=jnp.float32)
        loss, out = rebatch_call(
            module(config), rebatch=rebatch)(data)
        res_dict = {
            f"{name}_loss": item
            for name, item in out["losses"].items()
        }
        if "denoised_decoder" in out["results"]:
            res_dict["denoise"] = dict(marked=None, decoded=dict(
                residue_index=data["residue_index"][:1024],
                chain_index=data["chain_index"][:1024],
                batch_index=data["batch_index"][:1024],
                atom_pos=out["results"]["pos"],
                pos=out["results"]["pos"],
                aatype=data["aa_gt"][:1024],
                atom_mask=data["all_atom_mask"][:1024]
            ))
        return cast_float(loss, jnp.float32), cast_float(res_dict, jnp.float32)
    return step

def make_training_inner(optimizer, step, data, accumulate=1, multigpu=True, ema_weight=0.999, nanhunt=False):
    update = update_step(step, optimizer, accumulate=accumulate, multigpu=multigpu, nanhunt=False)
    def ema_step(params, loop_state):
        aux_state = loop_state.aux_state
        if "ema_params" not in loop_state.aux_state:
            aux_state = dict(ema_params=jax.tree_util.tree_map(lambda x: x, params))
        else:
            ema_params = loop_state.aux_state["ema_params"]
            ema_params = jax.tree_util.tree_map(lambda x, y: ema_weight * x + (1 - ema_weight) * y, ema_params, params)
            aux_state = dict(ema_params=ema_params)
        return aux_state
    def training_inner(loop_state: State):
        t = time.time()
        item = next(data)
        load_time = time.time() - t
        key, subkey = jax.random.split(loop_state.key)
        tr = time.time()
        aux_state = ema_step(loop_state.params, loop_state)
        _, loggables, params, opt_state = update(loop_state.params, loop_state.opt_state, subkey, item)
        step_time = time.time() - tr
        new_state = State(key, loop_state.step_id, params, opt_state, aux_state)
        checkpointables = {
            "checkpoint": params,
            "checkpoint-ema": aux_state["ema_params"]
        }
        total_time = time.time() - t
        print(f"Step {loop_state.step_id}, load time {load_time:.3f} s, step time {step_time:.3f} s, total {total_time:.3f} s")
        return new_state, loggables, checkpointables
    return training_inner

def cosine_decay_schedule(start_lr, decay_lr, warmup_steps, decay_steps):
    def schedule(count):
        result = jnp.where(
            count <= warmup_steps,
            count / warmup_steps * start_lr,
            (start_lr - decay_lr) * 0.5 * (1 + jnp.cos(jnp.pi * ((count - warmup_steps) % decay_steps) / decay_steps)) + decay_lr)
        result = jnp.maximum(result, 0.0)
        return result
    return schedule

def warmup_schedule(start_lr, warmup_steps):
    def schedule(count):
        result = jnp.minimum(count / warmup_steps * start_lr, start_lr)
        return result
    return schedule

def replicate_loop_state(state):
    devices = jax.devices()
    return State(
        state.key,
        state.step_id,
        jax.device_put_replicated(state.params, devices),
        jax.device_put_replicated(state.opt_state, devices),
        jax.device_put_replicated(state.aux_state, devices)
    )

def print_parameter_keys(params):
    def make_paths(x, prefix):
        if isinstance(x, dict):
            for key in x:
                for subkey in make_paths(x[key], prefix + "::" + key):
                    yield subkey
        yield prefix
    for key in make_paths(params, ""):
        print(key)

def log_decoded(writer, full_name: str, value, step):
    # FIXME: disable writing of debug PDB files
    if True:
        name = full_name.split("/")[-1]
        atom_mask = get_atom37_mask(value["aatype"])
        atom_mask *= value["atom_mask"].any(axis=1, keepdims=True)
        trp_type = np.full_like(value["aatype"], 17)
        trp_mask = get_atom37_mask(trp_type)
        trp_mask *= value["atom_mask"].any(axis=1, keepdims=True)
        if step % 100 == 0:
            protein = Protein(
                np.array(atom14_to_atom37(value["atom_pos"], value["aatype"])),
                np.array(value["aatype"]),
                np.array(atom_mask),
                np.array(value["residue_index"]),
                np.array(value["batch_index"]),
                np.array(atom_mask.astype(jnp.float32)))
            with open(f"{path}/{name}_{step}.pdb", "wt") as f:
                f.write(to_pdb(protein))

if __name__ == "__main__":
    from salad.data.allpdb import BatchedProteinPDBStream
    from flexloop.data import BatchStream

    opt = parse_options(
        "train a protein diffusion model on PDB.",
        path="network/",
        config="default",
        data_path="",
        num_aa=1024,
        p_complex=0.5,
        lr=1e-3,
        decay_lr=1e-5,
        warmup_steps=1_000,
        decay_steps=500_000,
        clip=0.1,
        b1=0.9,
        b2=0.99,
        ema_weight=0.999,
        rebatch=1,
        accumulate=1,
        jax_seed=42,
        afdb_max_L=0.4,
        afdb_min_plddt=70.0,
        multigpu="True",
        nanhunt="False",
        dataset="PDB",
        suffix="1",
        legacy_repetitive="True",
        schedule_fixed="False",
        order_agnostic="False",
        init_from="none",
    )
    NUM_DEVICES = jax.device_count()
    multigpu = opt.multigpu == "True"
    if opt.nanhunt == "True":
        multigpu = False
        NUM_DEVICES = 1
    num_aa = opt.num_aa
    if opt.dataset == "SYNTHETE":
        num_aa = 256
    elif opt.dataset.startswith("afdb"):
        num_aa = 1024
    path = opt.path
    path = f"{path}/salad_hybrid/simple-{opt.config}-{opt.num_aa}-{opt.suffix}"
    writer = SummaryWriter(path)

    print("Attempting to load dataset...")
    if opt.dataset == "SYNTHETE":
        from salad.data.synthete import SyntheteStream
        data = SyntheteStream(f"{opt.data_path}/synthete/")
    elif opt.dataset.startswith("afdb"):
        length_weights = False
        if "length" in opt.dataset:
            length_weights = True
        from salad.data.afdb import AFDBStream
        data = AFDBStream(
            f"{opt.data_path}/afdb/",
            f"{opt.data_path}/afdb/afdb1024_cluster_reps.csv",
            length_weights=length_weights,
            weight_max=500,
            min_size=32,
            # filter on loopiness and extent (fixed to the VP-scaled sampling range)
            filter=lambda x: (x["plddt"] > 85.0) * (x["L"] <= opt.afdb_max_L) * (x["std_ca"] <= 3 + x["length"] ** 0.4) > 0,
            min_plddt=opt.afdb_min_plddt,
            order_agnostic=opt.order_agnostic == "True")
    elif opt.dataset.startswith("mix"):
        from salad.data.afdb import AFDBStream
        from salad.data.mix_batch import MixBatch
        afdb_data = AFDBStream(
            f"{opt.data_path}/afdb/",
            f"{opt.data_path}/afdb/afdb1024_cluster_reps.csv",
            length_weights=False,
            weight_max=500,
            min_size=32,
            # filter on loopiness and extent (fixed to the VP-scaled sampling range)
            filter=lambda x: (x["plddt"] > 85.0) * (x["L"] <= opt.afdb_max_L) * (x["std_ca"] <= 3 + x["length"] ** 0.4) > 0,
            min_plddt=opt.afdb_min_plddt,
            order_agnostic=False)
        pdb_data = BatchedProteinPDBStream(f"{opt.data_path}/allpdb/",
                                           seqres_aa="clusterSeqresAA",
                                           cutoff_resolution=4.0,
                                           p_complex=opt.p_complex,
                                           size=opt.num_aa,
                                           min_size=16,
                                           max_size=opt.num_aa,
                                           legacy_repetitive_chains=False)
        data = MixBatch(((0.5, afdb_data), (0.5, pdb_data)))
    else:
        data = BatchedProteinPDBStream(f"{opt.data_path}/allpdb/",
                                        seqres_aa="clusterSeqresAA",
                                        cutoff_resolution=4.0,
                                        p_complex=opt.p_complex,
                                        size=opt.num_aa,
                                        min_size=16,
                                        max_size=opt.num_aa,
                                        legacy_repetitive_chains=opt.legacy_repetitive == "True")
    data = iter(BatchStream(data, num_workers=32,
                            accumulate=opt.rebatch * opt.accumulate * NUM_DEVICES,
                            prefetch_factor=32))
    print("Dataset successfully loaded.")

    config = getattr(config_choices, opt.config)
    key = jax.random.PRNGKey(opt.jax_seed)
    rebatch = opt.rebatch
    init, step = transformed = hk.transform(model_step(config, rebatch=rebatch, is_training=True))

    print("Initializing model parameters...")
    item_0 = next(data)
    print("INPUT OF SHAPE:")
    for name, value in item_0.items():
        print("  ", name, value.shape)
    init_batch = jax.tree_util.tree_map(lambda x: x[:rebatch * opt.accumulate * 100], item_0)
    params = init(key, init_batch)
    print("Model parameters initialized.")
    if opt.init_from != "none":
        print(f"Initializing parameters from checkpoint: {opt.init_from}...")
        import pickle
        param_path = opt.init_from
        with open(param_path, "rb") as f:
            override_params = pickle.load(f)
        for module_name in override_params:
            param_set = override_params[module_name]
            for param_name in param_set:
                params[module_name][param_name] = override_params[module_name][param_name]

    print("Writing model description...")
    tabulated = hk.experimental.tabulate(transformed)(init_batch)
    with open(f"{path}/model_description", "w") as f:
      f.write(tabulated)
    print("Model description written.")

    schedule = cosine_decay_schedule(
        start_lr=opt.lr, decay_lr=opt.decay_lr,
        warmup_steps=opt.warmup_steps, decay_steps=opt.decay_steps)
    total_steps = opt.warmup_steps + opt.decay_steps + 1
    if opt.schedule_fixed == "True":
        schedule = warmup_schedule(start_lr=opt.lr, warmup_steps=opt.warmup_steps)

    print("Initializing optimizer state...")
    optimizer = optax.chain(
        # clip gradients by their norm
        optax.clip_by_global_norm(opt.clip),
        # scale gradients using Adam
        optax.scale_by_adam(opt.b1, opt.b2, eps=1e-9),
        # scale resulting learning rate by a cosine schedule
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0))
    opt_state = optimizer.init(params)
    aux_state = {}
    print("Optimizer initialized.")

    print("Constructing training loop...")
    loop_state = State(key, 0, params, opt_state, aux_state)
    
    training_loop = training(
        path,
        make_training_inner(optimizer, step, data,
                            accumulate=opt.accumulate,
                            multigpu=multigpu,
                            ema_weight=opt.ema_weight),
        checkpoint_interval=10_000,
        max_steps=total_steps,
        logger=log(decoded=log_decoded))
    print("Recovering previous state, if available...")
    loop_state = load_loop_state(path) or loop_state
    print("Starting training...")
    print(f"Log files and tensorboard records will be written to {path}")
    training_loop(writer, loop_state)
