import time

from copy import deepcopy

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from torch.utils.tensorboard import SummaryWriter

from flexloop.simple_loop import (
    training, log, load_loop_state, update_step, rebatch_call, rebatch_call_bysize, State)
from salad.modules.gradient_diffusion import StructureDiffusion
from salad.modules.config import noise_schedule_benchmark as config_choices
from flexloop.utils import parse_options
from flexloop.loop import cast_float

def model_step(config, rebatch=1, is_training=True):
    module = StructureDiffusion
    if not is_training:
        config = deepcopy(config)
        config.eval = True
    def step(data):
        data = jax.tree_util.tree_map(lambda x: jnp.array(x), data)
        data = cast_float(data, dtype=jnp.float32)
        # FIXME: sharded model?
        # loss, out = rebatch_call(module(config), rebatch=rebatch)
        loss, out = rebatch_call(
            module(config), rebatch=rebatch)(data)
        res_dict = {
            f"{name}_loss": item
            for name, item in out["losses"].items()
        }
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

def replicate_loop_state(state):
    devices = jax.devices()
    return State(
        state.key,
        state.step_id,
        jax.device_put_replicated(state.params, devices),
        jax.device_put_replicated(state.opt_state, devices),
        jax.device_put_replicated(state.aux_state, devices)
    )

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
        multigpu="True",
        nanhunt="False",
        suffix="1"
    )
    NUM_DEVICES = jax.device_count()
    multigpu = opt.multigpu == "True"
    if opt.nanhunt == "True":
        multigpu = False
        NUM_DEVICES = 1
    path = opt.path
    path = f"{path}/salad/gd-{opt.config}-{opt.num_aa}-{opt.suffix}"
    writer = SummaryWriter(path)

    print("Attempting to load dataset...")
    data = BatchedProteinPDBStream(f"{opt.data_path}/allpdb/",
                                   seqres_aa="clusterSeqresAA",
                                   cutoff_resolution=4.0,
                                   p_complex=opt.p_complex,
                                   size=1024,
                                   min_size=16,
                                   max_size=1024)
    data = iter(BatchStream(data, num_workers=32,
                            accumulate=opt.rebatch * opt.accumulate * NUM_DEVICES,
                            prefetch_factor=32))
    print("Dataset successfully loaded.")

    config = getattr(config_choices, opt.config)
    key = jax.random.PRNGKey(opt.jax_seed)
    rebatch = opt.rebatch
    # FIXME: sharded rebatch?
    init, step = transformed = hk.transform(model_step(config, rebatch=rebatch, is_training=True))

    print("Initializing model parameters...")
    item_0 = next(data)
    print("INPUT OF SHAPE:")
    for name, value in item_0.items():
        print("  ", name, value.shape)
    init_batch = jax.tree_util.tree_map(lambda x: x[:rebatch * opt.accumulate * 100], item_0)
    params = init(key, init_batch)
    print("Model parameters initialized.")

    print("Writing model description...")
    tabulated = hk.experimental.tabulate(transformed)(init_batch)
    with open(f"{path}/model_description", "w") as f:
      f.write(tabulated)
    print("Model description written.")

    schedule = cosine_decay_schedule(
        start_lr=opt.lr, decay_lr=opt.decay_lr,
        warmup_steps=opt.warmup_steps, decay_steps=opt.decay_steps)
    total_steps = opt.warmup_steps + opt.decay_steps + 1

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
        max_steps=total_steps,
        logger=log())
    print("Recovering previous state, if available...")
    loop_state = load_loop_state(path) or loop_state
    # FIXME: explicit replication
    # if multigpu:
    #     loop_state = replicate_loop_state(loop_state)
    # FIXME: explicit replication
    # if multigpu:
    #     opt_state = jax.device_put_replicated(opt_state)
    print("Starting training...")
    print(f"Log files and tensorboard records will be written to {path}")
    training_loop(writer, loop_state)
