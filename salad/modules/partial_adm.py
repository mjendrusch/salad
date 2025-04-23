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
    SparseStructureAttention, SparseAttention,
    sequence_relative_position,
    distance_features,
    rotation_features,
    position_rotation_features,
    direction_features,
    pair_vector_features)
from salad.modules.utils.geometry import (
    extract_aa_frames, extract_neighbours, index_mean, compute_pseudo_cb, fast_sasa, distance_one_hot,
    get_spatial_neighbours, get_neighbours, get_random_neighbours, axis_index,
    get_index_neighbours)
from salad.modules.structure_autoencoder import InnerDistogram, assign_state
from salad.modules.config.distance_to_structure_decoder import small_vq
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

class PartialSequenceDiffusion(hk.Module):
    def __init__(self, config, name: Optional[str] = "partial_sequence_diffusion"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        data.update(self.prepare_data(data))
        data.update(self.prepare_diffusion(data))
        model = ADM(c)
        total, output = model(data)
        losses = output["losses"]
        return total, losses

    def prepare_data(self, data):
        c = self.config
        # truncate atom24 positions to atom14 position
        pos = data["all_atom_positions"][:, :14]
        atom_mask = data["all_atom_mask"][:, :14]
        aa_gt = data["aa_gt"]
        mask = atom_mask.any(axis=-1) * (aa_gt != 20)
        
        # assign dssp
        batch = data["batch_index"]
        dssp, *_ = assign_dssp(pos, batch, mask)

        # assign SASA
        sasa = fast_sasa(pos, atom_mask, aa_gt, batch)
        sasa = jnp.argmax(distance_one_hot(sasa, 0.0, 30.0, bins=16), axis=-1)

        # assign STATE
        state = assign_state(small_vq, c.param_path)(
            hk.next_rng_key(), data)

        return dict(
            pos=pos,
            atom_mask=atom_mask,
            mask=mask,
            dssp_gt=dssp,
            sasa_gt=sasa,
            state_gt=state
        )

    def prepare_diffusion(self, data):
        def drop_modality(x, batch, p_drop_all=0.1):
            beta = jax.random.beta(hk.next_rng_key(), 3, 9, (x.shape[0],))[batch]
            uniform = jax.random.uniform(hk.next_rng_key(), (x.shape[0],))[batch]
            use_beta = jax.random.bernoulli(hk.next_rng_key(), 0.2, shape=(x.shape[0],))[batch]
            p = use_beta * beta + (1 - use_beta) * uniform
            drop = jax.random.uniform(hk.next_rng_key(), (x.shape[0],)) < p
            drop_all = (jax.random.uniform(hk.next_rng_key(), shape=(x.shape[0],)) < p_drop_all)[batch]
            return drop * drop_all
        aa_gt = data["aa_gt"]
        dssp_gt = data["dssp_gt"]
        sasa_gt = data["sasa_gt"]
        state_gt = data["state_gt"]
        batch = data["batch_index"]
        atom_mask = data["atom_mask"]
        mask = data["mask"]
        corrupt_aa = drop_modality(aa_gt, batch) * mask
        aa = jnp.where(corrupt_aa, 20, aa_gt)
        corrupt_dssp = drop_modality(dssp_gt, batch) * mask
        dssp = jnp.where(corrupt_dssp, 3, dssp_gt)
        corrupt_sasa = drop_modality(sasa_gt, batch) * mask
        sasa = jnp.where(corrupt_sasa, 16, sasa_gt)
        corrupt_state = drop_modality(state_gt, batch, p_drop_all=0.25) * mask
        state = jnp.where(corrupt_state, 4096, state_gt)
        drop_pos = drop_modality(atom_mask, batch) * mask
        return dict(aa=aa, dssp=dssp, sasa=sasa, state=state, no_pos=drop_pos)

class ADM(hk.Module):
    def __init__(self, config, name: str | None = "partial_adm"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        stack = ADMStack(c)
        aa_head = MLP(c.local_size * 4, 20, activation=jax.nn.gelu, final_init=init_zeros())
        dssp_head = MLP(c.local_size * 4, 3, activation=jax.nn.gelu, final_init=init_zeros())
        sasa_head = MLP(c.local_size * 4, 16, activation=jax.nn.gelu, final_init=init_zeros())
        state_head = MLP(c.local_size * 4, 4096, activation=jax.nn.gelu, final_init=init_zeros())

        # prepare supervision neighbours
        pos = data["pos"][:, 1]
        batch = data["batch_index"]
        chain = data["chain_index"]
        resi = data["resi_index"]
        mask = data["mask"]
        resi_neighbours = get_index_neighbours(32)(
            resi, chain, batch, mask)
        pos_neighbours = get_spatial_neighbours(32)(
            Vec3Array.from_array(pos), batch, mask * (~data["no_pos"]))

        def model_iteration(data, prev):
            local, pos, resi, chain, batch, mask, pos_mask = self.prepare_features(data, prev)
            local = stack(local, pos, pos_mask,
                          resi_neighbours, pos_neighbours,
                          resi, chain, batch, mask)
            return local
        def iteration(i, prev):
            return model_iteration(data, prev)
        prev = jnp.zeros((data["aa"].shape[0], c.local_size), dtype=jnp.float32)
        if c.eval:
            count = c.num_recycles
        else:
            count = jax.random.randint(hk.next_rng_key(), (), 0, 3)
        prev = hk.fori_loop(0, count, iteration, prev)
        prev = jax.tree_map(jax.lax.stop_gradient, prev)
        local = model_iteration(data, prev)
        aa = aa_head(local)
        dssp = dssp_head(local)
        sasa = sasa_head(local)
        state = state_head(local)

        # mainline NLL losses
        mask = data["mask"]
        aa_predict_mask = (data["aa"] == 20) * (data["aa_gt"] != 20) * mask
        dssp_predict_mask = (data["dssp"] == 3) * (data["dssp_gt"] != 3) * mask
        sasa_predict_mask = (data["sasa"] == 16) * (data["sasa_gt"] != 16) * mask
        state_predict_mask = (data["state"] == 4096) * (data["state_gt"] != 4096) * mask
        aa_weight = aa_predict_mask / jnp.maximum(aa_predict_mask.sum(), 1)
        dssp_weight = dssp_predict_mask / jnp.maximum(dssp_predict_mask.sum(), 1)
        sasa_weight = sasa_predict_mask / jnp.maximum(sasa_predict_mask.sum(), 1)
        state_weight = state_predict_mask / jnp.maximum(state_predict_mask.sum(), 1)
        aa_nll = -(jax.nn.log_softmax(aa) * jax.nn.one_hot(data["aa_gt"], 20) * aa_predict_mask[..., None]).sum(axis=-1)
        aa_nll = (aa_nll * aa_weight).sum()
        dssp_nll = -(jax.nn.log_softmax(dssp) * jax.nn.one_hot(data["dssp_gt"], 3) * aa_predict_mask[..., None]).sum(axis=-1)
        dssp_nll = (dssp_nll * dssp_weight).sum()
        sasa_nll = -(jax.nn.log_softmax(sasa) * jax.nn.one_hot(data["sasa_gt"], 16) * aa_predict_mask[..., None]).sum(axis=-1)
        sasa_nll = (sasa_nll * sasa_weight).sum()
        state_nll = -(jax.nn.log_softmax(state) * jax.nn.one_hot(data["state_gt"], 16) * aa_predict_mask[..., None]).sum(axis=-1)
        state_nll = (state_nll * state_weight).sum()

        # prepare outputs
        out = dict(
            results=dict(
                aa=aa, corrupt_aa=aa_predict_mask, aa_gt=data["aa_gt"],
                dssp=dssp, corrupt_dssp=dssp_predict_mask, dssp_gt=data["dssp_gt"],
                sasa=sasa, corrupt_sasa=sasa_predict_mask, sasa_gt=data["sasa_gt"],
                state=state, corrupt_state=state_predict_mask, state_gt=data["state_gt"],),
            losses=dict(aa=aa_nll, dssp=dssp_nll, sasa=sasa_nll, state=state_nll)
                        #distogram=distogram_nll, distogram_trajectory=distogram_trajectory)
        )
        total = aa_nll + dssp_nll + sasa_nll + state_nll# + distogram_nll + distogram_trajectory
        return total, out
    
    def prepare_features(self, data, prev):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        current_aa = data["aa"]
        current_dssp = data["dssp"]
        current_sasa = data["sasa"]
        current_state = data["state"]
        prev_local = prev["local"]
        ncaco = data["all_atom_positions"][:, :4]
        cb = compute_pseudo_cb(ncaco)
        pos = jnp.concatenate((ncaco, cb[:, None]), axis=1)

        # when training, add random noise to backbone features
        if (not c.eval) or c.noise_eval:
            pos += c.noise_level * jax.random.normal(hk.next_rng_key(), pos.shape)
        pos = Vec3Array.from_array(pos)
        pos_mask = jnp.repeat(data["all_atom_mask"].any(axis=-1)[..., None], 5, axis=-1)
        pos_mask *= (~data["no_pos"])
        pos = pos.to_array()

        local = Linear(c.local_size * 2, initializer=init_linear(), bias=False)(
            jnp.concatenate((
                jax.nn.one_hot(current_aa, 21),
                jax.nn.one_hot(current_sasa, 17),
                jax.nn.one_hot(current_dssp, 4),
                jax.nn.one_hot(current_state, 4097)), axis=-1))
        local = Linear(c.local_size, initializer=init_linear(), bias=False)(
            jax.nn.gelu(local))
        local = hk.LayerNorm([-1], True, True)(local)
        local += dropout(MLP(c.local_size * 2, c.local_size, activation=jax.nn.gelu, final_init=init_linear())(
            hk.LayerNorm([-1], True, True)(prev_local)), c)
        return local, pos, resi, chain, batch, data["mask"], pos_mask.any(axis=-1) * (~data["no_pos"])

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

# class ADMBlock(hk.Module):
#     def __init__(self, config, name: Optional[str] = "adm_block"):
#         super().__init__(name)
#         self.config = config

#     def __call__(self, features, pos, pos_mask,
#                  resi, chain, batch,
#                  mask, sup_neighbours=None):
#         c = self.config
#         distogram = InnerDistogram(c)
#         distogram_logits, dmap = distogram(features, resi, chain, batch, sup_neighbours)
#         index = axis_index(sup_neighbours, axis=0)
#         resi_dist = abs(resi[:, None] - resi[None, :])
#         resi_neighbours = get_neighbours(32)(resi_dist, mask)
#         soft_neighbours = get_neighbours(32)(dmap, mask, resi_neighbours)
#         sup_distogram_logits = distogram_logits[index[:, None], sup_neighbours]
#         pair, pair_mask = soft_pair_features(c)(
#             distogram_logits, soft_neighbours, resi, chain, batch, mask)
#         features += SparseAttention(c)(
#             hk.LayerNorm([-1], True, True)(features),
#             pair, soft_neighbours, pair_mask)
#         pos_neighbours = get_spatial_neighbours(32)(
#             Vec3Array.from_array(pos), batch, mask * pos_mask)
#         pair, pair_mask = pos_pair_features(
#             pos, distogram_logits, pos_neighbours,
#             resi, chain, batch, pos_mask, mask)
#         pos_update = SparseStructureAttention(c)(
#             hk.LayerNorm([-1], True, True)(features),
#             pos, pair, pair_mask, pos_neighbours,
#             resi, chain, batch, mask)
#         features += jnp.where(pos_mask[..., None], pos_update, 0)
#         features += GlobalUpdate(c)(features, chain, batch, mask)
#         return features, sup_distogram_logits

class ADMBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "adm_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, pos_mask,
                 resi_neighbours, pos_neighbours,
                 resi, chain, batch,
                 mask):
        c = self.config
        pair, pair_mask = resi_pair_features(c)(
            features, resi, chain, batch, mask)
        features += SparseAttention(c)(
            hk.LayerNorm([-1], True, True)(features),
            pair, resi_neighbours, pair_mask)
        pair, pair_mask = pos_pair_features(
            pos, pos_neighbours,
            resi, chain, batch, pos_mask, mask)
        pos_update = SparseStructureAttention(c)(
            hk.LayerNorm([-1], True, True)(features),
            pos, pair, pair_mask, pos_neighbours,
            resi, chain, batch, mask)
        features += jnp.where(pos_mask[..., None], pos_update, 0)
        features += GlobalUpdate(c)(features, chain, batch, mask)
        return features

def soft_pair_features(c):
    def inner(logits, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        logits = logits[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, initializer="linear", bias=False)(
            jnp.where(pair_mask[..., None],
                      jax.nn.softmax(logits, axis=-1), 0))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

def resi_pair_features(c):
    def inner(features, neighbours,
              resi, chain, batch, mask):
        features = hk.LayerNorm([-1], True, True)(features)
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            features)[:, None]
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            features)[None, :]
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

def pos_pair_features(c):
    def inner(pos, logits, neighbours,
              resi, chain, batch, pos_mask, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        logits = logits[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, initializer="linear", bias=False)(
            jnp.where(pair_mask[..., None],
                      jax.nn.softmax(logits, axis=-1), 0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        pair = jnp.where(pos_mask[:, None, None], pair, 0)
        pair_mask *= pos_mask[:, None, None]
        return pair, pair_mask
    return inner

class GlobalUpdate(hk.Module):
    def __init__(self, config, name: str | None = "global"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, chain, batch, mask):
        c = self.config
        key = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer="relu")(local))
        key = key.reshape(*key.shape[:-1], c.global_heads, -1)
        value = Linear(local.shape[-1] * c.factor, initializer="linear")(local)
        value = value.reshape(*value.shape[:-1], c.global_heads, -1)
        query = Linear(local.shape[-1] * c.factor, initializer="relu")(local)
        query = query.reshape(*query.shape[:-1], c.global_heads, -1)
        query = jax.nn.gelu(query)
        bias = Linear(local.shape[-1] * c.factor, initializer="zeros")(local)
        bias = index_mean(bias, batch, mask[:, None])
        chain_operation = index_mean(
            key[:, :, :, None] * value[:, :, None, :], chain, mask[:, None, None, None])
        batch_operation = index_mean(
            key[:, :, :, None] * value[:, :, None, :], batch, mask[:, None, None, None])
        operation = jnp.concatenate((chain_operation, batch_operation), axis=-2)
        out = jnp.einsum("bhvk,bhk->bhv", operation, query).reshape(local.shape[0], -1) + bias
        out = Linear(local.shape[-1], bias=False, initializer="zeros")(out)
        return out

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

    def __call__(self, local, pos, pos_mask, resi, chain, batch,
                 neighbours, mask, sup_neighbours=None):
        c = self.config
        def diffusion_inner(block):
            def _inner(x):
                features = block(c)(x, pos, pos_mask,
                                    resi, chain, batch, neighbours,
                                    mask, sup_neighbours=sup_neighbours)
                if isinstance(features, tuple):
                    features, sup = features
                    return features, sup
                return features
            return _inner
        diffusion_block = ADMBlock
        stack = blocked_stack(
            c.depth, block_size=c.block_size, with_state=True)(
                hk.remat(diffusion_inner(diffusion_block)))
        local, distogram_trajectory = stack(local)
        local = hk.LayerNorm([-1], True, True)(local)
        if c.block_size > 1:
            distogram_trajectory = distogram_trajectory.reshape(
                -1, *distogram_trajectory.shape[2:])
        return local, distogram_trajectory
