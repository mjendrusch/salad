from typing import Optional
from functools import partial
from copy import deepcopy

import jax
import jax.numpy as jnp
import haiku as hk

# alphafold dependencies
from salad.aflib.model.geometry import Vec3Array
from salad.aflib.model.all_atom_multimer import get_atom14_mask

# basic module imports
from salad.modules.basic import (
    Linear, MLP, init_glorot, init_relu,
    init_zeros, init_linear, block_stack
)
from salad.modules.transformer import (
    resi_dual, prenorm_skip, resi_dual_input, prenorm_input, drop)

# import geometry utils
from salad.modules.utils.geometry import (
    index_mean, index_sum, index_count, extract_aa_frames,
    extract_neighbours, distance_rbf, hl_gaussian, distance_one_hot,
    unique_chain, positions_to_ncacocb, axis_index,
    single_protein_sidechains, compute_pseudo_cb,
    get_random_neighbours, get_spatial_neighbours,
    get_index_neighbours, get_neighbours, index_align)

from salad.modules.utils.dssp import assign_dssp, drop_dssp

from salad.modules.structure_autoencoder import (
    structure_augmentation, structure_augmentation_params,
    apply_structure_augmentation, apply_inverse_structure_augmentation,
    semiequivariant_update_positions,
    InnerDistogram, extract_dmap_neighbours)

# import violation loss
from salad.modules.utils.alphafold_loss import violation_loss

# sparse geometric module imports
from salad.modules.geometric import (
    SparseAttention,
    sequence_relative_position, distance_features,
    direction_features, type_position_features,
    position_rotation_features
)

from salad.modules.utils.diffusion import diffuse_sequence

class C2S(hk.Module):
    def __init__(self, config,
                 name: Optional[str] = "condition_to_sequence"):
        super().__init__(name)
        self.config = config

    def prepare_data(self, data):
        c = self.config
        pos = data["all_atom_positions"]
        atom_mask = data["all_atom_mask"]
        chain = data["chain_index"]
        batch = data["batch_index"]

        # recast positions into atom14:
        # first, truncate to atom14 format
        pos = pos[:, :14]
        atom_mask = atom_mask[:, :14]

        # uniquify chain IDs across batches:
        chain = unique_chain(chain, batch)
        mask = data["seq_mask"] * data["residue_mask"] * atom_mask[:, :3].all(axis=-1)
        
        # subtract the center from all positions
        center = index_mean(pos[:, 1], batch, atom_mask[:, 1, None])
        pos = pos - center[:, None]
        # set the positions of all masked atoms to the pseudo Cb position
        atom_pos = jnp.where(
            atom_mask[..., None], pos,
            compute_pseudo_cb(pos)[:, None, :])

        # transform data to ncacocb positions
        pos = positions_to_ncacocb(atom_pos)
        dssp, *_ = assign_dssp(pos, batch, mask)

        # add noise to positions
        if not c.eval:
            pos += c.noise_level * jax.random.normal(hk.next_rng_key(), pos.shape)

        result = dict(pos=pos, dssp=dssp, chain_index=chain, mask=mask)

        return result

    def prepare_condition(self, data):
        c = self.config
        mask = data["mask"]
        batch = data["batch_index"]
        pos = data["pos"]
        same_batch = batch[:, None] == batch[None, :]
        pair_mask = mask[:, None] * mask[None, :] * same_batch
        dist = jnp.linalg.norm(pos[:, None, 4] - pos[None, :, 4], axis=-1)
        contacts = dist < 8
        contacts *= pair_mask
        any_contacts = contacts.any(axis=1)
        cutoff = jax.random.uniform(hk.next_rng_key(), (), minval=0.0, maxval=12.0)
        contact_adjacent = (jnp.where(any_contacts[None, :], dist, jnp.inf) < cutoff).any(axis=1)
        contact_adjacent *= jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)
        dssp = data["dssp"]
        def make_mask(batch, p=c.p_condition if c.pair_condition is not None else 0.5, p_min=0.2, p_max=1.0):
            p_mask = jax.random.uniform(hk.next_rng_key(), shape=batch.shape, minval=p_min, maxval=p_max)[batch]
            bare_mask = jax.random.bernoulli(hk.next_rng_key(), p_mask)
            keep_mask = jax.random.bernoulli(hk.next_rng_key(), p=p, shape=batch.shape)[batch]
            return bare_mask * keep_mask

        dssp_mask = make_mask(batch)
        dssp = jnp.where(dssp_mask, dssp, 4)
        pos_mask = make_mask(batch)
        return dict(condition=dict(dssp=dssp,
                                   pos=pos, pos_mask=pos_mask,
                                   hotspots=contact_adjacent))

    def apply_diffusion(self, data):
        c = self.config
        chain = data["chain_index"]
        aa = data["aa_gt"]
        # use beta-linear task distribution, per chain
        t_beta = jax.random.beta(hk.next_rng_key(), 3, 9, aa.shape)
        t_linear = jax.random.uniform(hk.next_rng_key(), aa.shape)
        use_beta = jax.random.bernoulli(hk.next_rng_key(), 0.8)
        t_seq = jnp.where(use_beta, t_beta, t_linear)
        t_seq = t_seq[chain]
        if "t_seq" in data:
            t_seq = data["t_seq"]
        aa, corrupt_aa = diffuse_sequence(hk.next_rng_key(), aa, t_seq, mask_index=20)
        return dict(aa=aa, t_seq=t_seq, corrupt_aa=corrupt_aa)

    def __call__(self, data):
        c = self.config
        model = ADM(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # compute conditioning information
        data.update(self.prepare_condition(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))
        result = model(data)
        total, losses = model.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

# TODO
class C2SInference(C2S):
    def __init__(self, config,
                 name: Optional[str] = "condition_to_sequence"):
        super().__init__(config, name)

    def apply_diffusion(self, data):
        chain = data["chain_index"]
        aa = data["aa_gt"]
        # use beta-linear task distribution, per chain
        t_beta = jax.random.beta(hk.next_rng_key(), 3, 9, aa.shape)
        t_linear = jax.random.uniform(hk.next_rng_key(), aa.shape)
        use_beta = jax.random.bernoulli(hk.next_rng_key(), 0.8)
        t_seq = jnp.where(use_beta, t_beta, t_linear)
        t_seq = t_seq[chain]
        if "t_seq" in data:
            t_seq = data["t_seq"]
        aa, corrupt_aa = diffuse_sequence(hk.next_rng_key(), aa, t_seq, mask_index=20)
        return dict(aa=aa, t_seq=t_seq, corrupt_aa=corrupt_aa)

    def __call__(self, data):
        c = self.config
        model = ADM(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # initialize data
        data["aa"] = 20 * jnp.ones_like(data["aa_gt"])
        data["corrupt_aa"] = jnp.ones_like(data["aa_gt"], dtype=jnp.bool_)

        def sample_step(i, carry):
            aa, log_p = carry
            result = model(data)
            logits = result["aa"]
            probs = jax.nn.softmax(logits, axis=-1)
            highprobs = jnp.argsort(-probs, axis=-1)
            index = jnp.arange(probs.shape[0], dtype=jnp.int32)
            sortedprobs = probs[index[:, None], highprobs]
            p = c.temperature
            valid = (jnp.cumsum(sortedprobs, axis=-1) < p)
            probs = probs.at[index[:, None], highprobs].set(sortedprobs * valid)
            probs = probs / probs.sum(axis=-1, keepdims=True)
            logits = jnp.log(probs + 1e-6)
            entropy = (-logits * jax.nn.softmax(logits, axis=-1)).sum(axis=-1)
            entropy = jnp.where(aa == 20, jax.random.uniform(hk.next_rng_key(), aa.shape), jnp.inf)
            # entropy = jnp.where(data["mask"], entropy, jnp.inf)
            pos = jnp.argmin(entropy, axis=0)
            # jax.debug.print("pos {pos}", pos=pos)
            pos_aa = jax.random.categorical(hk.next_rng_key(), logits[pos], axis=-1)
            # jax.debug.print("posaa {pos_aa}", pos_aa=pos_aa)
            aa = aa.at[pos].set(pos_aa)
            log_p += jnp.where(data["mask"][pos], logits[pos, pos_aa], 0)
            return aa, log_p
        # jax.debug.print("val {val}", val=data["mask"].astype(jnp.int32).sum())
        aatype, log_p = hk.fori_loop(0, data["mask"].astype(jnp.int32).sum(), sample_step, (data["aa"], 0.0))
        # jax.debug.print("aatype {aatype}", aatype=aatype)
        # logits = model(data, model.init_prev(data))["aa"]
        # log_p_null = logits[jnp.arange(data["aa_gt"].shape[0]), data["aa_gt"]]
        # sequence_null = jnp.argmax(logits, axis=-1)
        # perplexity = jnp.exp(-(log_p_null * data["mask"]).sum() / jnp.maximum(data["mask"].sum(), 1))
        # recovery = ((aatype == data["aa_gt"]) * data["mask"]).sum() / jnp.maximum(data["mask"].sum(), 1)
        # recovery_null = ((sequence_null == data["aa_gt"]) * data["mask"]).sum() / jnp.maximum(data["mask"].sum(), 1)

        out = dict(
            aatype=aatype,
            log_p=log_p
        )
        return out

class ADM(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        stack = C2SStack(c, C2SBlock)
        features = self.prepare_features(data)
        local, pos, pos_mask, dssp, hotspot, neighbours, resi, chain, batch, mask = features
        # enable selective cyclisation of chains
        local = stack(
            local, pos, pos_mask, dssp, hotspot,
            neighbours, resi, chain, batch, mask)
        # predict aa
        aa = jax.nn.log_softmax(
            Linear(20, bias=False, initializer="zeros")(local), axis=-1)
        return dict(local=local, aa=aa)

    def prepare_features(self, data):
        c = self.config
        aa = data["aa"]
        if not c.continuous_aa:
            aa = jax.nn.one_hot(aa, 21, axis=-1)
        pos = data["condition"]["pos"]
        pos_mask = data["condition"]["pos_mask"]
        dssp = data["condition"]["dssp"]
        hotspot = data["condition"]["hotspots"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        neighbours = constant_neighbours(48)(
            pos, pos_mask, resi, chain, batch, mask)
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        relative_positions = (local_pos[:, :, None] - local_pos[:, None, :])
        relative_distance = distance_rbf(relative_positions.norm(), 0.0, 5.0, 16).reshape(batch.shape[0], -1)
        relative_direction = relative_positions.normalized().to_array().reshape(batch.shape[0], -1)
        local_pos = (local_pos / 5.0).to_array().reshape(batch.shape[0], -1)
        local = pos_mask[:, None] * Linear(c.local_size, bias=False, initializer="linear")(relative_distance)
        local += pos_mask[:, None] * Linear(c.local_size, bias=False, initializer="linear")(relative_direction)
        local += pos_mask[:, None] * Linear(c.local_size, bias=False, initializer="linear")(local_pos)
        local += Linear(c.local_size, bias=False, initializer="linear")(aa)
        local += Linear(c.local_size, bias=False, initializer="linear")(jax.nn.one_hot(dssp, 4, axis=-1))
        local += Linear(c.local_size, bias=False, initializer="linear")(hotspot.astype(jnp.float32)[:, None])
        local = hk.LayerNorm([-1], True, True)(local)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local)
        local = hk.LayerNorm([-1], True, True)(local)

        return local, pos, pos_mask, dssp, hotspot, neighbours, resi, chain, batch, mask

    def loss(self, data, result):
        mask = data["mask"]
        losses = dict()
        total = 0.0

        aa_predict_mask = mask * (data["aa_gt"] != 20) * data["corrupt_aa"]
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll
        total += aa_nll
        return total, losses

class C2SStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "encoder_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, pos_mask, dssp, hotspot,
                 neighbours, resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features = data
                features = block(c)(
                    features, pos, pos_mask, dssp, hotspot,
                    neighbours, resi, chain, batch, mask)
                return features
            return _inner
        diffusion_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=False)(
                hk.remat(stack_inner(diffusion_block)))
        if c.resi_dual:
            # handle ResiDual local features
            incremental = local
            local, incremental = stack(
                ((local, incremental)))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local = stack(local)
            local = hk.LayerNorm([-1], True, True)(local)
        return local

class C2SBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "encoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, pos_mask, dssp, hotspot,
                 neighbours, resi, chain, batch, mask):
        c = self.config
        def do_drop(x):
            if c.drop:
                return drop(x, p=0.1, is_training=not c.eval)
            return x
        residual_update, residual_input, _ = get_residual_gadgets(
            features, c.resi_dual)

        # we cannot use regular sparse IPA, as
        # most residues will not have position
        # information throughout training and eval.
        # Instead, we rely on our pair features
        # to provide richer attention biasing
        attention = SparseAttention
        aug_neighbours = random_neighbours()(
            neighbours, dssp, hotspot, chain, batch, mask)

        # constant neighbour attention
        pair, pair_mask = pos_pair_features(c)(
            residual_input(features),
            Vec3Array.from_array(pos), pos_mask,
            neighbours,
            resi, chain, batch, mask)
        pair = do_drop(pair)
        features = residual_update(
            features,
            do_drop(attention(c.key_size, c.heads)(
                residual_input(features), pair,
                neighbours, pair_mask)))
        # random neighbour attention
        pair, pair_mask = pos_pair_features(c)(
            residual_input(features),
            Vec3Array.from_array(pos), pos_mask,
            aug_neighbours,
            resi, chain, batch, mask)
        pair = do_drop(pair)
        features = residual_update(
            features,
            do_drop(attention(c.key_size, c.heads)(
                residual_input(features), pair,
                aug_neighbours, pair_mask)))
        # global update of local features
        features = residual_update(
            features,
            do_drop(Update(c)(
                residual_input(features),
                chain, batch, mask)))
        return features

def constant_neighbours(count=32, pos_count=16):
    def inner(pos, pos_mask, resi, chain, batch, mask):
        same_batch = batch[:, None] == batch[None, :]
        same_chain = chain[:, None] == chain[None, :]
        same_chain *= same_batch
        pair_mask = mask[:, None] * mask[None, :]
        pair_mask *= same_batch
        # residue-index based neighbours
        resi_dist = abs(resi[:, None] - resi[None, :])
        resi_dist = jnp.where(same_chain, resi_dist, jnp.inf)
        resi_dist = jnp.where(same_batch, resi_dist, jnp.inf)
        neighbours = get_neighbours(count)(resi_dist, pair_mask, None)
        # position based neighbours
        dist = jnp.linalg.norm(pos[:, None, 1] - pos[None, :, 1], axis=-1)
        dist_mask = pos_mask[:, None] * pos_mask[None, :] * pair_mask
        dist = jnp.where(dist_mask, dist, jnp.inf)
        neighbours = get_neighbours(pos_count)(dist, dist_mask, neighbours)
        return neighbours
    return inner

def random_neighbours(count=28, hotspot_count=4):
    def inner(neighbours, dssp, hotspot, chain, batch, mask):
        same_batch = batch[:, None] == batch[None, :]
        same_chain = chain[:, None] == chain[None, :]
        same_chain *= same_batch
        pair_mask = mask[:, None] * mask[None, :]
        pair_mask *= same_batch
        # beta sheets need to communicate
        beta = dssp == 2
        both_beta = beta[:, None] * beta[None, :] * pair_mask
        # different chains need to communicate
        other_chain = (1 - same_chain) > 0
        other_hotspot = other_chain * hotspot[None, :]
        # hotspot dist
        communicate = other_hotspot * pair_mask > 0
        dist = jnp.where(
            communicate,
            jax.random.uniform(hk.next_rng_key(), communicate.shape),
            1 + jax.random.uniform(hk.next_rng_key(), communicate.shape))
        neighbours = get_neighbours(hotspot_count)(
            dist, communicate, neighbours)
        # random additional neighbour distance
        communicate = (both_beta + same_batch) * pair_mask > 0
        dist = jnp.where(
            communicate,
            jax.random.uniform(hk.next_rng_key(), communicate.shape),
            jnp.inf)
        neighbours = get_neighbours(count)(
            dist, communicate, neighbours)[:, -(count + hotspot_count):]
        return neighbours
    return inner

def get_nearest_neighbours(num_neighbours):
    def inner(pos, batch, mask):
        if not isinstance(pos, Vec3Array):
            pos = Vec3Array.from_array(pos)
        same_batch = batch[:, None] == batch[None, :]
        cb = pos[:, 4]
        dist = (cb[:, None] - cb[None, :]).norm()
        dist = jnp.where(same_batch, dist, jnp.inf)
        return get_neighbours(num_neighbours)(dist, mask)
    return inner

def pos_pair_features(c):
    def inner(local, pos, pos_mask, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pos_pair_mask = pair_mask * pos_mask[:, None] * pos_mask[neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(local)[:, None]
        pair += Linear(c.pair_size, bias=False, initializer="linear")(local)[neighbours]
        pair += pos_pair_mask[..., None] * Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += pos_pair_mask[..., None] * Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += pos_pair_mask[..., None] * Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

def get_residual_gadgets(features, use_resi_dual=True):
    residual_update = (lambda x, y: resi_dual(*x, y)) if use_resi_dual else prenorm_skip
    residual_input = resi_dual_input if use_resi_dual else prenorm_input
    local_shape = features[0].shape[-1] if use_resi_dual else features.shape[-1]
    return residual_update, residual_input, local_shape

def make_pair_mask(mask, neighbours):
    return mask[:, None] * mask[neighbours] * (neighbours != -1)

class Update(hk.Module):
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, chain, batch, mask):
        c = self.config
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        chain_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        batch_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        hidden = batch_gate * index_mean(local_update, batch, mask[..., None])
        hidden += chain_gate * index_mean(local_update, chain, mask[..., None])
        hidden += local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(hidden)
        return result
