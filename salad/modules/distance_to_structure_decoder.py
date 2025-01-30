from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk

# alphafold dependencies
from alphafold.model.geometry import Vec3Array
from alphafold.model.all_atom_multimer import get_atom14_mask

# basic module imports
from salad.modules.basic import (
    Linear, MLP, init_glorot, init_relu,
    init_zeros, init_linear, block_stack
)
from salad.modules.transformer import (
    resi_dual, prenorm_skip, resi_dual_input, prenorm_input)

# import geometry utils
from salad.modules.utils.geometry import (
    index_mean, index_sum, index_count, extract_aa_frames,
    extract_neighbours, distance_rbf,
    unique_chain, positions_to_ncacocb,
    single_protein_sidechains, compute_pseudo_cb,
    get_random_neighbours, get_spatial_neighbours,
    get_neighbours, axis_index)

from salad.modules.utils.dssp import assign_dssp

# sparse geometric module imports
from salad.modules.geometric import (
    SparseStructureAttention, sequence_relative_position,
    distance_features, direction_features, pair_vector_features,
    position_rotation_features
)

class DistanceStructureDecoder(hk.Module):
    def __init__(self, config,
                 name: Optional[str] = "distance_structure_decoder"):
        super().__init__(name)
        self.config = config

    def prepare_data(self, data):
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
        pos = jnp.where(
            atom_mask[..., None], pos,
            compute_pseudo_cb(pos)[:, None, :])
        pos_14 = pos
        atom_mask_14 = atom_mask

        # get ncacocb positions
        pos_ncacocb = positions_to_ncacocb(pos)

        # make distance map
        cb = Vec3Array.from_array(pos_ncacocb[:, -1])
        # add noise
        cb += Vec3Array.from_array(0.3 * jax.random.normal(hk.next_rng_key(), list(cb.shape) + [3]))
        dmap = (cb[:, None] - cb[None, :]).norm()
        dmap_mask = batch[:, None] == batch[None, :]

        # set all-atom-position target
        atom_pos = pos_14
        atom_mask = atom_mask_14

        # set initial backbone positions
        pos = jax.random.normal(hk.next_rng_key(), pos_ncacocb.shape)
        return dict(pos=pos, pos_gt=pos_ncacocb,
                    dmap=dmap, dmap_mask=dmap_mask,
                    chain_index=chain, mask=mask,
                    atom_pos=atom_pos, atom_mask=atom_mask,
                    all_atom_positions=pos_14,
                    all_atom_mask=atom_mask_14)

    def __call__(self, data):
        c = self.config
        decoder = Decoder(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))

        # self-conditioning
        def iteration_body(i, prev):
            result = decoder(data, prev)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            return jax.lax.stop_gradient(prev)
        prev = dict(
            pos=data["pos"],
            local=jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32)
        )
        if not hk.running_init():
            if c.eval:
                count = c.num_recycle
            else:
                count = jax.random.randint(hk.next_rng_key(), (), 0, 4)
            prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = decoder(data, prev)
        total, losses = decoder.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict
    
class DistanceStructureDecoderInference(DistanceStructureDecoder):
    def __call__(self, data):
        c = self.config
        decoder = Decoder(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))

        # self-conditioning
        def iteration_body(i, prev):
            result = decoder(data, prev)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            return jax.lax.stop_gradient(prev)
        prev = dict(
            pos=data["pos"],
            local=jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32)
        )
        count = c.num_recycle
        prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = decoder(data, prev)
        out = dict(atom_pos=result["atom_pos"], aatype=jnp.argmax(result["aa"], axis=-1))
        return out

class AADecoderBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "aa_decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, features, pos,
                 neighbours, resi, chain, batch, mask):
        c = self.config
        # embed pair features
        pair, pair_mask = aa_decoder_pair_features(c)(
            Vec3Array.from_array(pos), neighbours, resi, chain, batch, mask)
        pair += hk.LayerNorm([-1], True, True)(
            Linear(pair.shape[-1], bias=False)(
                jax.nn.one_hot(aa, 21)))[neighbours]
        pair = MLP(
            2 * c.pair_size, c.pair_size, activation=jax.nn.gelu,
            final_init=init_linear())(pair)
        # attention / message passing module
        features += SparseStructureAttention(c)(
            hk.LayerNorm([-1], True, True)(features),
            pos, pair, pair_mask,
            neighbours, resi, chain, batch, mask)
        # local feature transition (always enabled)
        features += EncoderUpdate(c)(
                hk.LayerNorm([-1], True, True)(features), pos, chain, batch, mask)
        return features

class AADecoderStack(hk.Module):
    def __init__(self, config, depth=None, name: Optional[str] = "aa_decoder_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 3

    def __call__(self, aa, local, pos, neighbours,
                 resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                data = block(c)(aa, data, pos,
                                neighbours,
                                resi, chain,
                                batch, mask)
                return data
            return _inner
        stack = block_stack(
            self.depth, block_size=1, with_state=False)(
                hk.remat(stack_inner(AADecoderBlock)))
        local = hk.LayerNorm([-1], True, True)(stack(local))
        return local

class Decoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config
        self.decoder_block = DecoderBlock

    def __call__(self, data, prev):
        c = self.config
        decoder_stack = DecoderStack(c, self.decoder_block)
        aa_decoder = AADecoder(c)
        local, pos, dmap, resi, chain, batch, mask = self.prepare_features(data, prev)
        local, pos, trajectory = decoder_stack(
            local, pos, dmap, resi, chain, batch, mask)
        # predict & handle sequence, losses etc.
        result = dict()
        result["trajectory"] = trajectory
        result["local"] = local
        result["pos"] = pos
        # decoder features and logits
        aa_logits, decoder_features, corrupt_aa = aa_decoder.train(
            data["aa_gt"], local, pos, resi, chain, batch, mask)
        result["aa"] = aa_logits
        result["aa_features"] = decoder_features
        result["corrupt_aa"] = corrupt_aa * data["mask"] * (data["aa_gt"] != 20)
        result["aa_gt"] = data["aa_gt"]
        # generate all-atom positions using predicted
        # side-chain torsion angles:
        aatype = data["aa_gt"]
        if c.eval:
            aatype = jnp.argmax(aa_logits, axis=-1)
        raw_angles, angles, angle_pos = get_angle_positions(
            aatype, local, pos)
        result["raw_angles"] = raw_angles
        result["angles"] = angles
        result["atom_pos"] = angle_pos
        return result

    def init_prev(self, data):
        c = self.config
        return dict(
            pos = data["pos"],
            local = jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32)
        )

    def prepare_features(self, data, prev):
        c = self.config
        pos = prev["pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        dmap = data["dmap"]
        mask = data["mask"]
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1)
        ]
        local_features.append(hk.LayerNorm([-1], True, True)(prev["local"]))
        local_features = jnp.concatenate(
            local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
        local = hk.LayerNorm([-1], True, True)(local)

        return local, pos, dmap, resi, chain, batch, mask

    def loss(self, data, result):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        losses = dict()
        total = 0.0
        
        # AA NLL loss
        aa_predict_mask = mask * (data["aa_gt"] != 20)
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll
        total += c.aa_weight * aa_nll

        # position losses
        base_weight = mask / jnp.maximum(index_sum(mask.astype(jnp.float32), batch, mask), 1) / (batch.max() + 1)

        # sparse neighbour FAPE ** 2
        pair_mask = batch[:, None] == batch[None, :]
        pair_mask *= mask[:, None] * mask[None, :]
        pos_gt = data["pos_gt"]
        pos_gt = jnp.where(mask[:, None, None], pos_gt, 0)
        pos_gt = Vec3Array.from_array(pos_gt)
        frames_gt, _ = extract_aa_frames(jax.lax.stop_gradient(pos_gt))
        # CB distance
        distance = data["dmap"]
        distance = jnp.where(pair_mask, distance, jnp.inf)
        # get random neighbours to compute sparse FAPE on
        neighbours = get_random_neighbours(c.fape_neighbours)(distance, batch, mask)
        mask_neighbours = (neighbours != -1) * mask[:, None] * mask[neighbours]
        pos_gt_local = frames_gt[:, None, None].apply_inverse_to_point(pos_gt[neighbours])
        traj = Vec3Array.from_array(result["trajectory"])
        frames, _ = jax.vmap(extract_aa_frames)(traj)
        traj_local = frames[:, :, None, None].apply_inverse_to_point(traj[:, neighbours])
        fape_traj = (jnp.clip((traj_local - pos_gt_local).norm2(), 0.0, 100.0)).mean(axis=-1)
        fape_traj = jnp.where(mask_neighbours[None], fape_traj, 0)
        fape_traj = fape_traj.sum(axis=-1) / jnp.maximum(mask_neighbours.sum(axis=1)[None], 1)
        fape_traj = (fape_traj * base_weight).sum(axis=-1)
        losses["fape"] = fape_traj[-1] / 3
        losses["fape_trajectory"] = fape_traj.mean() / 3
        fape_loss = (c.fape_weight * fape_traj[-1] + c.fape_trajectory_weight * fape_traj.mean()) / 3
        total += fape_loss

        # local loss
        atom_pos = Vec3Array.from_array(result["atom_pos"])
        atom_pos_gt = Vec3Array.from_array(data["atom_pos"])
        local_neighbours = get_spatial_neighbours(
            count=c.local_neighbours)(
                pos_gt[:, -1], batch, mask)
        mask_gt = data["atom_mask"][local_neighbours]
        frames_gt, _ = extract_aa_frames(atom_pos_gt)
        atom_pos_gt = frames_gt[:, None, None].apply_inverse_to_point(atom_pos_gt[local_neighbours])
        frames, _ = extract_aa_frames(atom_pos)
        atom_pos = frames[:, None, None].apply_inverse_to_point(atom_pos[local_neighbours])
        local_loss = jnp.where(mask_gt, (atom_pos - atom_pos_gt).norm2(), 0).sum(axis=(1, 2))
        local_loss /= jnp.maximum(mask_gt.sum(axis=(1, 2)), 1)
        local_loss = (jnp.where(mask, local_loss, 0) * base_weight).sum() / 3
        losses["local"] = local_loss
        total += c.local_weight * local_loss

        return total, losses

def get_angle_positions(aa_gt, local, pos):
    frames, local_positions = extract_aa_frames(Vec3Array.from_array(pos))
    features = [
        local,
        local_positions.to_array().reshape(local_positions.shape[0], -1),
        distance_rbf(local_positions.norm(),
                     0.0, 10.0, 16).reshape(local_positions.shape[0], -1),
        jax.nn.one_hot(aa_gt, 21, axis=-1)
    ]
    raw_angles = MLP(
        local.shape[-1] * 2, 7 * 2, bias=False,
        activation=jax.nn.gelu, final_init="linear")(
            jnp.concatenate(features, axis=-1))

    raw_angles = raw_angles.reshape(-1, 7, 2)
    angles = raw_angles / jnp.sqrt(jnp.maximum(
        (raw_angles ** 2).sum(axis=-1, keepdims=True), 1e-6))
    angle_pos, _ = single_protein_sidechains(
        aa_gt, frames, angles)
    angle_pos = angle_pos.to_array().reshape(-1, 14, 3)
    angle_pos = jnp.concatenate((
        pos[..., :4, :],
        angle_pos[..., 4:, :]
    ), axis=-2)
    return raw_angles, angles, angle_pos

class AADecoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "aa_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, local, pos, resi, chain, batch, mask):
        c = self.config
        neighbours = get_spatial_neighbours(32)(
            Vec3Array.from_array(pos)[:, -1], batch, mask)
        local = AADecoderStack(c, depth=c.aa_decoder_depth)(
            aa, local, pos, neighbours,
            resi, chain, batch, mask)
        local = hk.LayerNorm([-1], True, True)(local)
        return Linear(20, initializer=init_zeros(), bias=False)(local), local

    def train(self, aa, local, pos, resi, chain, batch, mask):
        c = self.config
        aa = 20 * jnp.ones_like(aa)
        logits, features = self(
            aa, local, pos, resi, chain, batch, mask)
        logits = jax.nn.log_softmax(logits)
        return logits, features, jnp.ones_like(mask)

class DecoderStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "decoder_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, dmap,
                 resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the decoder block
                features, pos = block(c)(
                    features, pos, dmap,
                    resi, chain, batch, mask)
                trajectory_output = pos
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), trajectory_output
            return _inner
        decoder_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(decoder_block)))
        (local, pos), trajectory = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        return local, pos, trajectory

def aa_decoder_pair_features(c):
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        return pair, pair_mask
    return inner

def decoder_pair_features(c):
    def inner(pos, dmap, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        dmap = dmap[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, initializer="linear", bias=False)(
            jnp.where(pair_mask[..., None],
                      distance_rbf(dmap), 0))
        pos = Vec3Array.from_array(pos)
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
        return pair, pair_mask
    return inner

def extract_dmap_neighbours(count=32):
    def inner(distance, resi, chain, batch, mask):
        same_item = batch[:, None] == batch[None, :]
        weight = -3 * jnp.log(distance + 1e-6)
        uniform = jax.random.uniform(hk.next_rng_key(), weight.shape, dtype=weight.dtype, minval=1e-6, maxval=1 - 1e-6)
        gumbel = jnp.log(-jnp.log(uniform))
        weight = weight - gumbel
        distance = -weight
        distance = jnp.where(same_item, distance, jnp.inf)
        mask = (mask[:, None] * mask[None, :] * same_item)
        return get_neighbours(count)(distance, mask)
    return inner

def extract_neighbours(num_index=16, num_spatial=16, num_random=16):
    def inner(pos, resi, chain, item, mask):
        pos = pos[:, 1]
        same_batch = (item[:, None] == item[None, :])
        same_chain = (chain[:, None] == chain[None, :])
        valid = same_batch * (mask[:, None] * mask[None, :])
        within = abs(resi[:, None] - resi[None, :]) < num_index
        within *= same_batch * same_chain
        distance = (pos[:, None] - pos[None, :]).norm()
        distance = jnp.where(within, jnp.inf, distance)
        distance = jnp.where(valid, distance, jnp.inf)
        cutoff = jnp.sort(distance)[:, :num_spatial][:, -1]
        within = within + (distance < cutoff) > 0
        random_distance = -3 * jnp.log(jnp.maximum(distance, 1e-6))
        gumbel = jax.random.gumbel(hk.next_rng_key(), random_distance.shape)
        random_distance = jnp.where(within, -10_000, -(random_distance - gumbel))
        random_distance = jnp.where(valid, random_distance, jnp.inf)
        neighbours = get_neighbours(
            num_index + num_spatial + num_random)(
                random_distance, mask)
        return neighbours
    return inner

class DecoderBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, dmap,
                 resi, chain, batch, mask):
        c = self.config
        current_neighbours = extract_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=32)(
                Vec3Array.from_array(pos),
                resi, chain, batch, mask)
        dmap_neighbours = extract_dmap_neighbours(
            count=64)(
                dmap, resi, chain, batch, mask)

        pair, pair_mask = decoder_pair_features(c)(
            pos, dmap, current_neighbours,
            resi, chain, batch, mask)
        features += SparseStructureAttention(c)(
                hk.LayerNorm([-1], True, True)(features),
                pos / c.sigma_data, pair, pair_mask,
                current_neighbours, resi, chain, batch, mask)
        pair, pair_mask = decoder_pair_features(c)(
            pos, dmap, dmap_neighbours,
            resi, chain, batch, mask)
        features += SparseStructureAttention(c)(
                hk.LayerNorm([-1], True, True)(features),
                pos / c.sigma_data, pair, pair_mask,
                dmap_neighbours, resi, chain, batch, mask)
        # global update of local features
        features += DecoderUpdate(c)(
                hk.LayerNorm([-1], True, True)(features),
                pos, chain, batch, mask)
        local_norm = hk.LayerNorm([-1], True, True)(features)
        # update positions
        split_scale_out = c.sigma_data
        pos = update_positions(pos, local_norm,
                               scale=split_scale_out,
                               symm=c.symm)
        return features, pos.astype(local_norm.dtype)

def make_pair_mask(mask, neighbours):
    return mask[:, None] * mask[neighbours] * (neighbours != -1)

class EncoderUpdate(hk.Module):
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, chain, batch, mask):
        c = self.config
        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        local_pos = local_pos.to_array().reshape(local_pos.shape[0], -1)
        local += MLP(local.shape[-1] * 2,
                     local.shape[-1],
                     activation=jax.nn.gelu,
                     final_init=init_zeros())(local_pos)
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        local_local = local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(local_local)
        return result

class DecoderUpdate(hk.Module):
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, chain, batch, mask):
        c = self.config
        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        local_pos = local_pos.to_array().reshape(local_pos.shape[0], -1)
        local += MLP(local.shape[-1] * 2,
                     local.shape[-1],
                     activation=jax.nn.gelu,
                     final_init=init_zeros())(local_pos)
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        chain_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        batch_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        hidden = index_mean(batch_gate * local_update, batch, mask[..., None])
        hidden += index_mean(chain_gate * local_update, chain, mask[..., None])
        hidden += local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(hidden)
        return result

def update_positions(pos, local_norm, scale=10.0, symm=None):
    frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
    pos_update = scale * Linear(
        pos.shape[-2] * 3, initializer=init_zeros(),
        bias=False)(local_norm)
    # potentially symmetrize position update
    if symm is not None:
        pos_update = symm(pos_update)
    pos_update = Vec3Array.from_array(
        pos_update.reshape(*pos_update.shape[:-1], -1, 3))
    local_pos += pos_update

    # project updated pos to global coordinates
    pos = frames[..., None].apply_to_point(local_pos).to_array()
    return pos
