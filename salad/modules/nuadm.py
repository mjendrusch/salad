from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk

from salad.aflib.model.layer_stack import layer_stack
from salad.aflib.model.all_atom_multimer import atom37_to_atom14
from salad.aflib.model.geometry import Vec3Array

from salad.modules.basic import (
    Linear, MLP, init_glorot, init_linear, init_zeros)
from salad.modules.geometric import (
    SparseStructureAttention, SparseStructureMessage,
    sequence_relative_position,
    distance_features,
    rotation_features,
    pair_vector_features)
from salad.modules.utils.geometry import (
    extract_aa_frames, extract_neighbours, index_mean, compute_pseudo_cb)
from salad.modules.utils.dssp import assign_dssp

def rbf_embedding(distance, min_distance=0.0, max_distance=22.0, bins=64):
    step = (max_distance - min_distance) / bins
    centers = min_distance + jnp.arange(bins) * step + step / 2
    rbf = jnp.exp(-((distance[..., None] - centers) / step) ** 2)
    return rbf

def bond_angle(x, y, z):
    left = x - y
    right = z - y
    cos_tau = (left * right).sum(axis=-1) / jnp.maximum(jnp.linalg.norm(left, axis=-1) * jnp.linalg.norm(right, axis=-1), 1e-6)
    return jnp.arccos(cos_tau) / jnp.pi * 180

def dihedral_angle(a, b, c, d):
    x = b - a
    y = c - b
    z = d - c
    y_norm = jnp.linalg.norm(y, axis=-1)
    result = jnp.arctan2(y_norm * (x * jnp.cross(y, z)).sum(axis=-1),
                         (jnp.cross(x, y) * jnp.cross(y, z)).sum(axis=-1))
    return result / jnp.pi * 180

def compute_backbone_dihedrals(pos, resi, chain, batch):
    bb = pos[:, :3, :].reshape(-1, 3)
    allowed = resi[1:] == resi[:-1] + 1
    allowed *= chain[1:] == chain[:-1]
    allowed *= batch[1:] == batch[:-1]
    allowed = jnp.repeat(allowed, 3)
    allowed = jnp.concatenate((allowed, jnp.zeros((3,))), axis=0)
    dihedrals = dihedral_angle(bb[:-3], bb[1:-2], bb[2:-1], bb[3:])
    dihedrals = jnp.concatenate((dihedrals, jnp.zeros((3,))), axis=0)
    dihedrals = dihedrals.reshape(pos.shape[0], 3)
    dihedrals *= jnp.pi / 180
    return jax.lax.stop_gradient(dihedrals)

class NuADM(hk.Module):
    def __init__(self, config, name: Optional[str] = "nu_adm"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        data.update(self.prepare_diffusion(data))
        stack = ADMStack(c)
        aa_predictor = MLP(c.local_size * 4, 20, activation=jax.nn.gelu, final_init=init_zeros())
        def update_aa(aa, aa_gt):
            masked = aa == 20
            update = jax.random.bernoulli(hk.next_rng_key(), jnp.where(masked, 0.05, 0.0))
            return jnp.where(update, aa_gt, aa)
        def model_iteration(data, prev):
            local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data, prev)
            local = stack(local, pos, resi, chain, batch, neighbours, mask)
            aa = jnp.argmax(aa_predictor(local), axis=-1)
            return local, dict(
                local=local,
                aa=aa
            )
        def iteration(i, prev):
            return model_iteration(data, prev)[1]
        prev = dict(
            aa=data["aa"],
            local=jnp.zeros((data["aa"].shape[0], c.local_size), dtype=jnp.float32)
        )
        # FIXME
        # count = 0 # jax.random.randint(hk.next_rng_key(), (), 0, 2)
        # prev = hk.fori_loop(0, count, iteration, prev)
        # prev = jax.tree_map(jax.lax.stop_gradient, prev)
        local, _ = model_iteration(data, prev)
        aa = aa_predictor(local)
        predict_mask = (data["aa"] == 20) * (data["aa_gt"] != 20) * data["mask"]
        batch = data["batch_index"]
        pmask_count = jnp.zeros_like(predict_mask, dtype=jnp.float32).at[batch].add(predict_mask)[batch]
        aa_count = jnp.zeros_like(predict_mask, dtype=jnp.float32).at[batch].add(data["mask"])[batch]
        if c.unweighted:
            weight = predict_mask / jnp.maximum(predict_mask.sum(), 1)
        else:
            weight = predict_mask * 1 / jnp.maximum(pmask_count, 1) / (batch.max() + 1) # / data["mask"].sum()
        nll = -(jax.nn.log_softmax(aa) * jax.nn.one_hot(data["aa_gt"], 20) * predict_mask[..., None]).sum(axis=-1)
        nll = (nll * weight).sum()
        out = dict(
            results=dict(aa=aa, corrupt_aa=predict_mask, aa_gt=data["aa_gt"]),
            losses=dict(aa=nll)
        )
        return nll, out

    def prepare_diffusion(self, data):
        # truncate atom24 positions to atom14 position
        pos = data["all_atom_positions"][:, :14]
        atom_mask = data["all_atom_mask"][:, :14]
        aa_gt = data["aa_gt"]
        mask = atom_mask.any(axis=-1) * (aa_gt != 20)
        p = jax.random.uniform(hk.next_rng_key(), (mask.shape[0],))[data["batch_index"]]
        corrupt_mask = jax.random.bernoulli(hk.next_rng_key(), p)
        if self.config.mask_all:
            corrupt_mask = jnp.ones_like(corrupt_mask)
        aa = jnp.where(corrupt_mask, 20, aa_gt)
        return dict(aa=aa, all_atom_positions=pos, all_atom_mask=atom_mask, mask=mask)

    def prepare_features(self, data, prev):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        current_aa = data["aa"]
        prev_aa = prev["aa"]
        prev_local = prev["local"]
        ncaco = data["all_atom_positions"][:, :4]
        cb = compute_pseudo_cb(ncaco)
        pos = jnp.concatenate((ncaco, cb[:, None]), axis=1)
        clean_pos = pos

        # assign secondary structure elements
        sse, _, _ = assign_dssp(pos, batch, data["all_atom_mask"].any(axis=-1))

        # when training, add random noise to backbone features
        if (not c.eval) or c.noise_eval:
            pos += c.noise_level * jax.random.normal(hk.next_rng_key(), pos.shape)
        pos = Vec3Array.from_array(pos)
        pos_mask = jnp.repeat(data["all_atom_mask"].any(axis=-1)[..., None], 5, axis=-1)
        dihedrals = compute_backbone_dihedrals(pos.to_array(), resi, chain, batch)
        neighbours = extract_neighbours(
            c.num_index,
            c.num_spatial,
            c.num_random)(pos, resi, chain, batch, pos_mask.any(axis=-1))
        same_chain = chain[:, None] == chain[neighbours]
        resi_distance = jnp.clip(resi[:, None] - resi[neighbours], -32, 32) + 32
        resi_distance = jnp.where(same_chain, resi_distance, 64)
        resi_distance = jax.nn.one_hot(resi_distance, 65)
        backbone_distance = rbf_embedding((pos[:, None, :, None] - pos[neighbours, None, :]).norm(), 0.0, 22.0, 16)
        backbone_distance = backbone_distance.reshape(*neighbours.shape, -1)
        frames, _ = extract_aa_frames(pos)
        relative_rotation = (frames.rotation[:, None].inverse() @ frames.rotation[neighbours]).to_array()
        relative_rotation = relative_rotation.reshape(*neighbours.shape, -1)
        direction = frames.translation[neighbours] - frames.translation[:, None]
        direction = direction.normalized().to_array()
        pos = pos.to_array()

        # backbone_dihedrals = ... # TODO
        # backbone_angles = ... # TODO
        frequency_info = (dihedrals[..., None] * (2 ** jnp.arange(5))).reshape(dihedrals.shape[0], -1)
        local = Linear(c.local_size, initializer=init_glorot())(
            jnp.concatenate((
                jax.nn.one_hot(current_aa, 21),
                jax.nn.one_hot(prev_aa, 21),
                jnp.sin(frequency_info),
                jnp.cos(frequency_info),
                # backbone_angles,
                jax.nn.one_hot(sse, 3)), axis=-1))
        neighbour_mask = neighbours != -1
        pair = Linear(c.local_size, initializer=init_glorot())(
            jnp.concatenate((
                resi_distance,
                backbone_distance,
                relative_rotation,
                direction
            ), axis=-1)
        )
        pair = MLP(c.local_size * 2, c.local_size, final_init=init_glorot())(pair)
        weight = MLP(c.local_size * 2, 1, final_init=init_zeros())(pair)
        weight = jax.nn.softmax(jnp.where(neighbour_mask[..., None], weight, -jnp.inf), axis=1)
        weight = jnp.where(neighbour_mask[..., None], weight, 0)
        local += dropout((pair * weight).sum(axis=1), c)

        local = hk.LayerNorm([-1], True, True)(local)
        local += dropout(MLP(c.local_size * 2, c.local_size, activation=jax.nn.gelu, final_init=init_glorot())(
            hk.LayerNorm([-1], True, True)(prev_local)), c)
        if c.noise_once:
            clean_pos = pos
        return local, clean_pos, neighbours, resi, chain, batch, pos_mask.any(axis=-1)
    
class NuADMInference(hk.Module):
    def __init__(self, config, name: Optional[str] = "nu_adm"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        data.update(self.prepare_data(data))
        stack = ADMStack(c)
        aa_predictor = MLP(c.local_size * 4, 20, activation=jax.nn.gelu, final_init=init_zeros())

        def model_iteration(data, prev):
            tie_index = data["tie_index"]
            local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data, prev)
            local = stack(local, pos, resi, chain, batch, neighbours, mask)
            aa_logits = jax.nn.log_softmax(aa_predictor(local), axis=-1)
            normalization = jnp.zeros_like(tie_index).at[tie_index].add(1)[tie_index]
            aa_logits = jnp.zeros_like(aa_logits).at[tie_index].add(aa_logits)[tie_index] / jnp.maximum(normalization[:, None], 1)
            aa_logits /= data["temperature"][:, None]
            aa_logits = jax.nn.log_softmax(aa_logits, axis=-1)
            aa_sample = jax.random.categorical(hk.next_rng_key(), aa_logits, axis=-1)
            aa_sample = aa_sample.at[tie_index].set(aa_sample)[tie_index]
            is_masked = data["aa"] == 20
            change_position = jax.random.uniform(hk.next_rng_key(), is_masked.shape) * is_masked
            change_position = jax.nn.one_hot(jnp.argmax(change_position, axis=0), change_position.shape[0], dtype=jnp.bool_)
            change_position = change_position.at[tie_index].max(change_position)[tie_index]
            new_aa = jnp.where(change_position, aa_sample, data["aa"])
            return local, dict(
                local=prev["local"],
                aa=new_aa
            )
        def iteration(i, carry):
            data, prev = carry
            _, update = model_iteration(data, prev)
            data["aa"] = update["aa"]
            prev["aa"] = update["aa"]
            return (data, prev)

        prev = dict(
            aa=data["aa"],
            local=jnp.zeros((data["aa"].shape[0], c.local_size), dtype=jnp.float32)
        )
        data, prev = hk.fori_loop(0, data["aa"].shape[0], iteration, (data, prev))
        # data, _ = iteration(0, (data, prev))
        return data["aa"]

    def prepare_data(self, data):
        atom_positions = data["all_atom_positions"]
        atom_mask = data["all_atom_mask"]
        if atom_mask.shape[1] == 24:
            # truncate atom24 positions to atom14 position
            pos = atom_positions[:, :14]
            atom_mask = atom_mask[:, :14]
        elif atom_mask.shape[1] == 14:
            pos = atom_positions
            atom_mask = atom_mask
        elif atom_mask.shape[1] == 37:
            pos, atom_mask = atom37_to_atom14(
                jnp.zeros_like(data["aa"]),
                Vec3Array.from_array(atom_positions),
                atom_mask)
            atom_mask = jnp.ones_like(atom_mask)
            pos = pos.to_array()
        mask = atom_mask.any(axis=-1)
        return dict(all_atom_positions=pos, all_atom_mask=atom_mask, mask=mask)

    def prepare_features(self, data, prev):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        current_aa = data["aa"]
        prev_aa = prev["aa"]
        prev_local = prev["local"]
        ncaco = data["all_atom_positions"][:, :4]
        cb = compute_pseudo_cb(ncaco)
        pos = jnp.concatenate((ncaco, cb[:, None]), axis=1)
        clean_pos = pos

        # assign secondary structure elements
        sse, _, _ = assign_dssp(pos, batch, data["all_atom_mask"].any(axis=-1))

        # when training, add random noise to backbone features
        if (not c.eval) or c.noise_eval:
            pos += c.noise_level * jax.random.normal(hk.next_rng_key(), pos.shape)
        pos = Vec3Array.from_array(pos)
        pos_mask = jnp.repeat(data["all_atom_mask"].any(axis=-1)[..., None], 5, axis=-1)
        dihedrals = compute_backbone_dihedrals(pos.to_array(), resi, chain, batch)
        neighbours = extract_neighbours(
            c.num_index,
            c.num_spatial,
            c.num_random)(pos, resi, chain, batch, pos_mask.any(axis=-1))
        same_chain = chain[:, None] == chain[neighbours]
        resi_distance = jnp.clip(resi[:, None] - resi[neighbours], -32, 32) + 32
        resi_distance = jnp.where(same_chain, resi_distance, 64)
        resi_distance = jax.nn.one_hot(resi_distance, 65)
        backbone_distance = rbf_embedding((pos[:, None, :, None] - pos[neighbours, None, :]).norm(), 0.0, 22.0, 16)
        backbone_distance = backbone_distance.reshape(*neighbours.shape, -1)
        frames, _ = extract_aa_frames(pos)
        relative_rotation = (frames.rotation[:, None].inverse() @ frames.rotation[neighbours]).to_array()
        relative_rotation = relative_rotation.reshape(*neighbours.shape, -1)
        direction = frames.translation[neighbours] - frames.translation[:, None]
        direction = direction.normalized().to_array()
        pos = pos.to_array()

        # backbone_dihedrals = ... # TODO
        # backbone_angles = ... # TODO
        frequency_info = (dihedrals[..., None] * (2 ** jnp.arange(5))).reshape(dihedrals.shape[0], -1)
        local = Linear(c.local_size, initializer=init_glorot())(
            jnp.concatenate((
                jax.nn.one_hot(current_aa, 21),
                jax.nn.one_hot(prev_aa, 21),
                jnp.sin(frequency_info),
                jnp.cos(frequency_info),
                # backbone_angles,
                jax.nn.one_hot(sse, 3)), axis=-1))
        neighbour_mask = neighbours != -1
        pair = Linear(c.local_size, initializer=init_glorot())(
            jnp.concatenate((
                resi_distance,
                backbone_distance,
                relative_rotation,
                direction
            ), axis=-1)
        )
        pair = MLP(c.local_size * 2, c.local_size, final_init=init_glorot())(pair)
        weight = MLP(c.local_size * 2, 1, final_init=init_zeros())(pair)
        weight = jax.nn.softmax(jnp.where(neighbour_mask[..., None], weight, -jnp.inf), axis=1)
        weight = jnp.where(neighbour_mask[..., None], weight, 0)
        local += dropout((pair * weight).sum(axis=1), c)

        local = hk.LayerNorm([-1], True, True)(local)
        local += dropout(MLP(c.local_size * 2, c.local_size, activation=jax.nn.gelu, final_init=init_glorot())(
            hk.LayerNorm([-1], True, True)(prev_local)), c)
        return local, clean_pos, neighbours, resi, chain, batch, pos_mask.any(axis=-1)

class ADMBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "adm_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, incremental, pos,
                 residue_index, chain_index, batch_index,
                 neighbours, mask):
        c = self.config
        update_init = c.update_init if c.update_init else init_zeros()
        sparse_structure_block = SparseStructureAttention
        if (not (c.eval or c.noise_once)) or c.noise_eval:
            pos = pos + c.noise_level * jax.random.normal(hk.next_rng_key(), pos.shape)
        if c.global_update:
            local, incremental = resi_dual(
                local, incremental, 
                GlobalUpdate(c, final_init=update_init)(
                    local, chain_index, batch_index, mask), c)
        pos = Vec3Array.from_array(pos)
        pair = Linear(c.pair_size, bias=False, initializer=init_linear())(
            sequence_relative_position(32, True)(
                residue_index, chain_index, batch_index, neighbours, ))
        pair += Linear(c.pair_size, bias=False, initializer=init_linear())(
            distance_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer=init_linear())(
            rotation_features(extract_aa_frames(pos)[0], neighbours))
        pair += Linear(c.pair_size, bias=False, initializer=init_linear())(
            pair_vector_features(pos, neighbours, scale=0.1))
        pair_mask = mask[:, None] * mask[neighbours] * (neighbours != -1)
        pos = pos.to_array()
        local, incremental = resi_dual(
            local, incremental,
            sparse_structure_block(c, normalize=False)(
                local, pos, pair, pair_mask, neighbours,
                residue_index, chain_index, batch_index, mask), c)
        local, incremental = resi_dual(
            local, incremental,
            MLP(c.factor * local.shape[-1], local.shape[-1],
                final_init=update_init, activation=jax.nn.gelu)(local), c)
        return local, incremental

class GlobalUpdate(hk.Module):
    def __init__(self, factor, final_init="zeros",
                 name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.factor = factor
        self.final_init = final_init

    def __call__(self, local, chain, batch, mask):
        local_update = Linear(local.shape[-1] * 2, initializer="relu")(local)
        local_batch = jax.nn.gelu(index_mean(local_update, batch, mask[..., None]))
        local_chain = jax.nn.gelu(index_mean(local_update, chain, mask[..., None]))
        result = Linear(local.shape[-1], initializer=self.final_init)(jnp.concatenate((local_batch, local_chain), axis=-1))
        return result

def resi_dual(local, incremental, output, config):
    if not config.eval:
        mask = jax.random.bernoulli(hk.next_rng_key(), 0.1, output.shape)
        output = jnp.where(mask, 0, output) / 0.9
    local = hk.LayerNorm([-1], True, True)(local + output)
    incremental = incremental + output
    return local, incremental

def dropout(data, config):
    if not config.eval:
        mask = jax.random.bernoulli(hk.next_rng_key(), 0.1, data.shape)
        data = jnp.where(mask, 0, data) / 0.9
    return data

def blocked_stack(depth, block_size=1, with_state=False):
    count = depth // block_size
    def inner(function):
        if block_size > 1:
            block = hk.remat(layer_stack(block_size, with_state=with_state)(function))
            return layer_stack(count, with_state=with_state)(block)
        return layer_stack(count, with_state=with_state)(function)
    return inner

class ADMStack(hk.Module):
    def __init__(self, config, name: Optional[str] = "adm_stack"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, offset, resi, chain, batch, neighbours, mask):
        c = self.config
        incremental = local
        def diffusion_inner(block):
            def _inner(x):
                local, incremental = x
                local, incremental = block(c)(local, incremental, offset,
                                              resi, chain, batch, neighbours,
                                              mask)
                return local, incremental
            return _inner
        diffusion_block = ADMBlock
        stack = blocked_stack(c.depth, block_size=c.block_size, with_state=False)(hk.remat(diffusion_inner(diffusion_block)))
        result = (local, incremental)
        result = stack(result)
        local, incremental = result
        local = local + hk.LayerNorm([-1], True, True)(incremental)        
        return local
