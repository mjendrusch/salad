from torch.multiprocessing import set_start_method, get_start_method
if __name__ == "__main__":
  set_start_method('spawn', force=True)
  print("START METHOD", get_start_method())

import pickle
import random
from typing import Any

import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard.writer import SummaryWriter

def rebatch_data(data, count):
  return jax.tree_util.tree_map(lambda x: x.reshape(count, -1, *x.shape[1:]), data)

def debatch_output(loss, out):
  def debatch_inner(x):
    if len(x) == 1:
      return x.mean()
    if len(x) > 1:
      return x[0]
  loss = loss.mean()
  out = jax.tree_util.tree_map(debatch_inner, out)
  out["losses"] = jax.tree_util.tree_map(lambda x: x.mean(), out["losses"])
  return loss, out

def train_step(config, rebatch=1, is_training=True):
  module = Hallucination
  def step(data):
    data = jax.tree_util.tree_map(lambda x: jnp.array(x), data)
    data = cast_float(data, dtype=jnp.float32)
    fold = module(config)
    if rebatch > 1:
      fold = hk.vmap(fold, split_rng=(not hk.running_init()))
      rebatched = rebatch_data(data, rebatch)
      loss, out = fold(rebatched)
      loss, out = debatch_output(loss, out)
    else:
      loss, out = fold(data)
      loss = loss.mean()
    res_dict = {
      f"{name}_loss": item
      for name, item in out["losses"].items()
    }

    aa_probs = jax.nn.softmax(out["results"]["aa"], axis=-1)
    confusion = jnp.zeros((21, 20), dtype=jnp.float32)
    confusion = confusion.at[out["results"]["aa_gt"]].add(
      aa_probs * out["results"]["corrupt_aa"][..., None])
    confusion /= jnp.maximum(confusion.sum(axis=1, keepdims=True), 1e-3)
    confusion = confusion[:20, :20]
    res_dict["confusion"] = loggable("confusion", confusion)

    mean_dist = (jax.nn.softmax(out["results"]["distogram"], axis=-1) * jnp.arange(64)).sum(axis=-1)
    res_dict["distogram"] = loggable("distogram", mean_dist)

    out_ca = out["results"]["atom_pos"][:512, 1]
    out_dist = jnp.linalg.norm(
      out_ca[:, None] - out_ca[None, :], axis=-1)
    out_dist = jnp.clip(out_dist, 0, 22) / 22 * 64
    res_dict["direct_distances"] = loggable(
        "distogram", out_dist
    )
    # out_ca = data["all_atom_positions"][:1600, 1]
    # out_dist = jnp.linalg.norm(out_ca[:, None] - out_ca[None, :], axis=-1)
    # out_dist = jnp.clip(out_dist, 0, 22) / 22 * 64
    # res_dict["gt_distances"] = loggable(
    #     "distogram", out_dist
    # )
    # out_ca = out["results"]["pos_noised"][:1600, 1]
    # out_dist = jnp.linalg.norm(
    #     out_ca[:, None] - out_ca[None, :], axis=-1)
    # out_dist = jnp.clip(out_dist, 0, 22) / 22 * 64
    # res_dict["noised_distances"] = loggable(
    #     "distogram", out_dist
    # )

    return cast_float(loss, jnp.float32), cast_float(res_dict, jnp.float32)
  return step

def ema_step(gamma=0.999):
  def inner(params, opt_state, aux_state, key, item, loop=None):
    full_params = params
    if isinstance(params, tuple):
      params = full_params[0]
    if "ema_params" not in aux_state:
      aux_state["ema_params"] = params
    else:
      aux_state["ema_params"] = jax.tree_util.tree_map(
        lambda x, y: x * gamma + (1 - gamma) * y,
        aux_state["ema_params"], params
      )
    if loop.step_id % 1000 == 0:
      loop.checkpoint.checkpoint(aux_state["ema_params"], f"ema-{loop.step_id}")
    return full_params, opt_state, aux_state
  return inert_step(inner)

def cosine_decay(target=1e-3, steps=10_000, decay_steps=100_000):
  def schedule(count):
    result = jnp.where(
        count <= steps,
        count / steps * target,
        target * 0.5 ** ((count - steps) // decay_steps) * 0.5 * (1 + jnp.cos(jnp.pi * ((count - steps) % decay_steps) / decay_steps)))
    result = jnp.maximum(result, 0.0)
    return result
  return schedule

def log_confusion(writer: SummaryWriter, name, value, step):
  if step % 10 == 0:
    value = np.array(value)[None, :, :]
    writer.add_image(name, value, step)

def log_histogram(writer: SummaryWriter, name, value, step):
  if step % 100 == 0:
    value = np.array(value)
    writer.add_histogram(name, value, step, bins=np.linspace(0.0, 1.0, 11))

def log_distogram(writer: SummaryWriter, name, value, step):
  if step % 100 == 0:
    value = np.array(value)[None, :, :] / 64
    writer.add_image(name, value, step)

class NoneData(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return None

    def __len__(self):
        return 10000

class SingleBatchData(Dataset):
  def __init__(self, data, batch_size):
    self.batch = [random.choice(data) for idx in range(batch_size)]

  def __getitem__(self, index) -> Any:
    return self.batch[index]

  def __len__(self):
    return len(self.batch)

def filter_res(x):
    return x["resolution"] <= 3.5

class AccumulateWrapper(Dataset):
  def __init__(self, base, factor=1):
    self.base = base
    self.factor = factor

  def next(self) -> Any:
    items = []
    for idx in range(self.factor):
      items.append(self.base.next())
    return {
      "train": {
        name: np.concatenate(list(map(lambda x: x["train"][name], items)), axis=0)
        for name in items[0]["train"]
      },
      "ema": items[0]["ema"]
    }

  def __len__(self) -> int:
    return len(self.base) // self.factor

if __name__ == "__main__":
    set_start_method("spawn", force=True)
    # FIXME: move this outside again
    import jax
    import jax.numpy as jnp
    import optax
    import haiku as hk

    from flexloop.loop import (
      Checkpoint, Log, loggable, TrainingLoop, inert_step, cast_float
    )
    from flexloop.data import BatchDistribution
    from flexloop.utils import parse_options
    from salad.data.allpdb import ProteinCropPDB

    import salad.modules.config.structure_diffusion as configuration
    from salad.modules.obsolete.protein_hallucination import Hallucination

    NUM_DEVICES = len(jax.devices())
    opt = parse_options(
        "Train unconditional protein diffusion with JAX.",
        path="./network",
        config="default",
        rebatch=1,
        new_block="False",
        data_path=".",
        batch_size=128,
        diffusion_scale=50.0,
        warmup_steps=10_000,
        decay_steps=100_000,
        accumulate=1,
        num_aa=128,
        mode="default",
        use_sigma="False",
        schedule="default",
        multigpu="False",
        single_batch="False",
        multiprocess="True",
        lion="False",
        multimer="False",
        tf_schedule="False",
        time_weighting="correct",
        double="False",
        p_complex=0.1,
        fine_tune="none",
        suffix="1",
        clip=0.1,
        lr=1e-4,
        b1=0.9,
        b2=0.999
    )
    path = f"{opt.path}/salad/latentdiff-allpdb-{opt.config}-{opt.num_aa}" \
           f"-{opt.lr}-{opt.suffix}" 
    config = getattr(configuration, opt.config)
    config.new_block = opt.new_block == "True"
    if opt.new_block == "True":
        config.block_type = "edm"
    train = train_step(config, rebatch=opt.rebatch)

    msazero_data = ProteinCropPDB(
        f"{opt.data_path}/allpdb/",
        opt.num_aa,
        cutoff_date="12/31/20", # FIXME: end in 2020
        cutoff_resolution=3.5,
        seqres_aa="clusterSeqresAA",
        seqres_na="clusterSeqresNA",
        assembly=False,
    )

    if opt.single_batch == "True":
        msazero_data = SingleBatchData(msazero_data, 2 * opt.batch_size)

    tasks = [
        ("train", msazero_data),
        ("ema", NoneData())
    ]
    tasks = [
        ("train", msazero_data),
        ("ema", NoneData())
    ]
    data: BatchDistribution = AccumulateWrapper(BatchDistribution(
        tasks,
        persistent_workers=True,
        prefetch_factor=16,
        num_workers=16
    ), factor=NUM_DEVICES * opt.accumulate * opt.rebatch)

    BATCH_FACTOR=NUM_DEVICES * opt.accumulate * opt.rebatch
    init_batch = data.next()["train"]
    init_batch = jax.tree_util.tree_map(lambda x: x[:BATCH_FACTOR * 10], init_batch)
    rng = jax.random.PRNGKey(42)
    transformed = hk.transform(train_step(config, rebatch=opt.rebatch))
    init, step = transformed
    _, valid_step = hk.transform(train_step(config, rebatch=1, is_training=False))

    params = init(rng, init_batch)

    per_item_clipping = None
    schedule = cosine_decay(
        opt.lr, steps=opt.warmup_steps, decay_steps=opt.decay_steps)
    optimizer_step = optax.scale_by_adam(opt.b1, opt.b2, eps=1e-9)
    lr_scale = -1.0
    optimizer = optax.chain(
        optax.clip_by_global_norm(opt.clip),
        optimizer_step,
        # optax.add_decayed_weights(0.01, True),
        optax.scale_by_schedule(schedule),
        optax.scale(lr_scale)
    )
    opt_state = optimizer.init(params)
    aux_state = {}

    train = TrainingLoop(
        log=Log(path).add(
          confusion=log_confusion,
          distogram=log_distogram,
          histogram=log_histogram
        ),
        checkpoint=Checkpoint(path),
        optimizer=optimizer,
        per_item_transform=per_item_clipping,
        checkpoint_interval=1000,
        valid_interval=1000,
        with_state=config.resample_loss,
        multigpu=opt.multigpu == "True",
        accumulate=opt.accumulate,
    )
    tabulated = hk.experimental.tabulate(transformed)(init_batch)
    with open(f"{path}/model_description", "w") as f:
      f.write(tabulated)

    params, opt_state, aux_state, rng = train.load(
        params, opt_state, aux_state, rng
    )
    if opt.fine_tune != "none":
      with open(opt.fine_tune, "rb") as f:
        params = pickle.load(f)
    train.steps["train"] = step
    train.steps["ema"] = ema_step(0.999)
    train.valid_steps["valid"] = valid_step

    train.train(params, opt_state, rng, data,
                valid=None, aux_state=aux_state)
