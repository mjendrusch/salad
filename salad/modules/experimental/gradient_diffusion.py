from typing import Optional
from functools import partial

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
    resi_dual, prenorm_skip, resi_dual_input, prenorm_input)

# import geometry utils
from salad.modules.utils.geometry import (
    index_mean, index_sum, index_count, extract_aa_frames,
    extract_neighbours, distance_rbf, hl_gaussian, distance_one_hot,
    unique_chain, positions_to_ncacocb, axis_index,
    single_protein_sidechains, compute_pseudo_cb,
    get_random_neighbours, get_spatial_neighbours,
    get_index_neighbours, get_neighbours, index_align)

from salad.modules.utils.dssp import assign_dssp, drop_dssp

# TODO add pre Encoder and post Encoder augmentation to semi-equivariant mode
from salad.modules.structure_autoencoder import (
    structure_augmentation, structure_augmentation_params,
    apply_structure_augmentation, apply_inverse_structure_augmentation,
    semiequivariant_update_positions,
    InnerDistogram, extract_dmap_neighbours)

# import violation loss
from salad.modules.utils.alphafold_loss import violation_loss

# sparse geometric module imports
from salad.modules.geometric import (
    SparseStructureAttention,
    SemiEquivariantSparseStructureAttention,
    VectorLinear,
    vector_mean_norm,
    sequence_relative_position, distance_features,
    direction_features, pair_vector_features,
    position_rotation_features
)

# diffusion processes
from salad.modules.utils.diffusion import (
    diffuse_coordinates_edm, diffuse_atom_cloud,
    diffuse_sequence, fourier_time_embedding
)

def sigma_scale_cosine(time, s=0.01):
    s = 0.01
    alpha_bar = jnp.cos((time + s) / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar = jnp.clip(alpha_bar, 0, 1)
    sigma = jnp.sqrt(1 - alpha_bar)
    return sigma

def sigma_scale_framediff(time, bmin=0.1, bmax=20.0):
    Gs = time * bmin + 0.5 * time ** 2 * (bmax - bmin)
    sigma = jnp.sqrt(1 - jnp.exp(-Gs))
    return sigma

def sigma_scale_none(time):
    return time

SIGMA_SCALE = dict(
    cosine=sigma_scale_cosine,
    framediff=sigma_scale_framediff,
    none=sigma_scale_none,
    linear=sigma_scale_none
)

class StructureDiffusion(hk.Module):
    def __init__(self, config,
                 name: Optional[str] = "structure_diffusion"):
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
        pos = jnp.where(
            atom_mask[..., None], pos,
            compute_pseudo_cb(pos)[:, None, :])
        # if running a semi-equivariant model, augment structures
        if c.equivariance == "semi_equivariant":
            pos = structure_augmentation(
                pos / c.sigma_data, data["batch_index"], data["mask"]) * c.sigma_data
        pos_14 = pos
        atom_mask_14 = atom_mask

        # if specified, augment backbone with encoded positions
        data["chain_index"] = chain
        data["pos"] = pos
        data["atom_mask"] = atom_mask
        data["mask"] = mask
        seq, pos = self.encode(data)

        # set all-atom-position target
        atom_pos = pos_14
        atom_mask = atom_mask_14
        return dict(seq=seq, pos=pos, chain_index=chain, mask=mask,
                    atom_pos=atom_pos, atom_mask=atom_mask,
                    all_atom_positions=pos_14,
                    all_atom_mask=atom_mask_14)

    def encode(self, data):
        pos = data["pos"]
        pos_14 = pos
        aatype = data["aa_gt"]
        c = self.config
        # set up ncaco + pseudo cb positions
        backbone = positions_to_ncacocb(pos)
        pos = backbone
        if c.equivariance == "semi_equivariant":
            center, translation, rotation = structure_augmentation_params(
                pos / c.sigma_data, data["batch_index"], data["mask"])
            pos = c.sigma_data * apply_structure_augmentation(
                pos / c.sigma_data, center, translation, rotation)
        data["pos"] = pos
        if c.augment_size > 0:
            # add additional vector features
            _, augmented_pos = Encoder(c)(data)
            pos = augmented_pos
        elif c.encode_atom14:
            pos = pos_14
            if c.equivariance == "semi_equivariant":
                center, translation, rotation = structure_augmentation_params(
                    pos / c.sigma_data, data["batch_index"], data["mask"])
                pos = c.sigma_data * apply_structure_augmentation(
                    pos / c.sigma_data, center, translation, rotation)
        else:
            pos = backbone
        if c.equivariance == "semi_equivariant":
            pos = c.sigma_data * apply_inverse_structure_augmentation(
                pos / c.sigma_data, center, translation, rotation)
        seq = aatype
        return seq, pos

    def prepare_condition(self, data):
        c = self.config
        aa = data["aa_gt"]
        pos = data["atom_pos"]
        pos_mask = data["atom_mask"]
        mask = data["mask"]
        batch = data["batch_index"]
        chain = data["chain_index"]
        same_batch = batch[:, None] == batch[None, :]

        def get_masks(chain_index, batch_index, set_condition=None):
            if set_condition is not None:
                sequence_mask = set_condition["aa"] != 20
                sse_mask = set_condition["dssp"] != 3
                sse_mean_mask = set_condition["dssp_mean_mask"]
                return sequence_mask, sse_mask, sse_mean_mask
            def mask_me(batch, p=0.5, p_min=0.2, p_max=1.0):
                p_mask = jax.random.uniform(hk.next_rng_key(), shape=batch.shape, minval=p_min, maxval=p_max)[batch]
                bare_mask = jax.random.bernoulli(hk.next_rng_key(), p_mask)
                keep_mask = jax.random.bernoulli(hk.next_rng_key(), p=p, shape=batch.shape)[batch]
                return bare_mask * keep_mask
            sequence_mask = mask_me(batch_index, p=0.5)
            sse_mask = mask_me(batch_index, p=0.5)
            sse_mean_mask = mask_me(chain_index, p=0.5, p_min=0.8)
            return sequence_mask, sse_mask, sse_mean_mask

        dssp, blocks, block_adjacency = assign_dssp(pos, data["batch_index"], pos_mask.any(axis=-1))
        _, block_adjacency = drop_dssp(
            hk.next_rng_key(), dssp, blocks, block_adjacency, p_drop=c.p_drop_dssp)
        sequence_mask, dssp_mask, dssp_mean_mask = get_masks(chain, batch)
        def aa_loss(aa_pred, use_condition):
            aa_mask = (aa != 20) * sequence_mask * mask * use_condition
            aa_gt = jax.nn.one_hot(aa, 20, axis=-1)
            weight = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
            return (aa_mask * weight * (jax.nn.log_softmax(aa_pred, axis=-1) * aa_gt).sum(axis=-1)).sum()
        def dssp_loss(dssp_pred, use_condition):
            d_mask = (dssp != 4) * dssp_mask * mask * use_condition
            dssp_gt = jax.nn.one_hot(dssp, 3, axis=-1)
            weight = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
            return (d_mask * weight * (jax.nn.log_softmax(dssp_pred, axis=-1) * dssp_gt).sum(axis=-1)).sum()
        dssp_mean = index_mean(jax.nn.one_hot(dssp, 3, axis=-1), chain, mask[:, None])
        def dssp_mean_loss(dssp_pred, use_condition):
            condition_mask = dssp_mean_mask * mask * use_condition
            dssp_pred = jax.nn.softmax(dssp_pred)
            dssp_pred = index_mean(dssp_pred, chain, mask[:, None])
            dssp_pred = jnp.log(dssp_pred + 1e-6)
            weight = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
            result = (jax.nn.log_softmax(dssp_pred, axis=-1) * dssp_mean).sum(axis=-1)
            result /= jnp.maximum(index_count(chain, condition_mask), 1)
            result = (weight * condition_mask * result).sum()
            return result
        # set up pair condition
        ca = Vec3Array.from_array(data["pos"][:, 1])
        dmap = (ca[:, None] - ca[None, :]).norm()
        percentage = jax.random.uniform(hk.next_rng_key(), (ca.shape[0],), minval=0.2, maxval=0.8)[batch]
        struc_mask = jax.random.bernoulli(hk.next_rng_key(), percentage)
        struc_mask *= jax.random.bernoulli(hk.next_rng_key(), p=0.5, shape=batch.shape)[chain]
        dmap_mask = struc_mask[:, None] * struc_mask[None, :] * same_batch
        def pos_loss(pos, use_condition):
            ca = pos
            ca = jnp.where(use_condition[:, None], ca, jax.lax.stop_gradient(ca))
            ca = Vec3Array.from_array(ca)
            ca_map = (ca[:, None] - ca[None, :]).norm()
            weight = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
            result = (dmap_mask * weight[:, None] * (ca_map - dmap) ** 2).sum()
            return -result
        def fitness(pos, aa_pred, dssp_pred):
            # include conditioning stochastically during training
            use_condition = jax.random.bernoulli(hk.next_rng_key(), 0.5, (aa.shape[0],))
            use_condition = use_condition[data["batch_index"]]
            pos_val = pos_loss(pos, use_condition)
            aa_val = aa_loss(aa_pred, use_condition)
            dssp_val = dssp_loss(dssp_pred, use_condition)
            dssp_mean_val = dssp_mean_loss(dssp_pred, use_condition)
            dssp_val = jnp.where(
                jax.random.bernoulli(hk.next_rng_key(), 0.5),
                dssp_val, dssp_mean_val)
            return pos_val + aa_val + dssp_val

        return fitness, dssp

    def apply_diffusion(self, data):
        c = self.config
        batch = data["batch_index"]
        mask = data["mask"]
        pos = Vec3Array.from_array(data["pos"])
        seq = data["seq"]
        if c.diffusion_kind == "edm":
            sigma_pos, _ = get_sigma_edm(
                batch,
                meanval=c.pos_mean_sigma,
                stdval=c.pos_std_sigma,
                minval=c.pos_min_sigma,
                maxval=c.pos_max_sigma)
            sigma_seq = jax.random.uniform(
                hk.next_rng_key(), (batch.shape[0],))[batch]
            if "t_pos" in data:
                t_pos = data["t_pos"]
                t_seq = data["t_seq"]
            pos_noised = diffuse_coordinates_edm(
                hk.next_rng_key(), pos, batch, sigma_pos[:, None])
            seq_noised, corrupt_aa = diffuse_sequence(
                hk.next_rng_key(), seq, sigma_seq)
            t_pos = sigma_pos
            t_seq = sigma_seq
        if c.diffusion_kind in ("vp", "vpfixed", "flow", "flow_scaled"):
            t_pos = jax.random.uniform(
                hk.next_rng_key(), (batch.shape[0],))[batch]
            sigma = SIGMA_SCALE[c.diffusion_time_scale](t_pos)
            t_pos = sigma
            t_seq = jax.random.uniform(
                hk.next_rng_key(), (batch.shape[0],))[batch]
            if "t_pos" in data:
                t_pos = data["t_pos"]
                t_seq = data["t_seq"]
            cloud_std = None
            if c.diffusion_kind in ("vpfixed", "flow"):
                cloud_std = c.sigma_data
            pos_noised = diffuse_atom_cloud(
                hk.next_rng_key(), pos, mask, batch,
                t_pos[:, None], cloud_std=cloud_std,
                flow=c.diffusion_kind in ("flow", "flow_scaled"))
            seq_noised, corrupt_aa = diffuse_sequence(
                hk.next_rng_key(), seq, t_seq)
        return dict(
            seq_noised=seq_noised, pos_noised=pos_noised.to_array(),
            t_pos=t_pos, t_seq=t_seq, corrupt_aa=corrupt_aa)

    def __call__(self, data):
        c = self.config
        diffusion = Diffusion(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # prepare condition / task information
        fitness, dssp_gt = self.prepare_condition(data)
        data["dssp_gt"] = dssp_gt
        # apply noise to data
        data.update(self.apply_diffusion(data))

        # self-conditioning
        def iteration_body(i, prev):
            override = self.apply_diffusion(data)
            result = diffusion(data, prev, fitness, override=override)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            return jax.lax.stop_gradient(prev)
        prev = diffusion.init_prev(data)
        if not hk.running_init():
           count = jax.random.randint(hk.next_rng_key(), (), 0, 2)
           prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = diffusion(data, prev, fitness)
        total, losses = diffusion.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

class StructureDiffusionInference(StructureDiffusion):
    def __init__(self, config,
                 name: Optional[str] = "structure_diffusion"):
        super().__init__(config, name)
        self.config = config

    def __call__(self, data, prev, fitness):
        c = self.config
        diffusion = Diffusion(c)
        # apply noise to data
        data.update(self.apply_diffusion(data))
        # potentially process noised data
        if c.sym_noise is not None:
            data["pos_noised"] = c.sym_noise(data["pos_noised"])
        def apply_model(prev):
            result = diffusion(data, prev, fitness, predict_aa=True)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            return result, prev

        if prev is None:
            prev = diffusion.init_prev(data)
        result, prev = apply_model(prev)

        # NOTE: also return noisy input for diagnostics
        result["pos_input"] = data["pos_noised"]
        if c.aa_trajectory:
            result["aa"] = result["aa_trajectory"][-1]
            result["aatype"] = jnp.argmax(result["aa_trajectory"][-1], axis=-1)

        return result, prev

    def apply_diffusion(self, data):
        c = self.config
        batch = data["batch_index"]
        mask = data["mask"]
        pos = data["pos"]

        # FIXME: center inputs
        # pos = pos - index_mean(pos[:, 1], batch, mask[:, None])[:, None]

        pos = Vec3Array.from_array(pos)
        seq = data["seq"]
        t_pos = data["t_pos"]
        t_seq = data["t_seq"]


        if c.diffusion_kind == "edm":
            pos_noised = diffuse_coordinates_edm(
                hk.next_rng_key(), pos, batch, t_pos[:, None])
            seq_noised, corrupt_aa = diffuse_sequence(
                hk.next_rng_key(), seq, t_seq)
        if c.diffusion_kind in ("vp", "vpfixed", "flow"):
            cloud_std = None
            if c.diffusion_kind in ("vpfixed", "flow"):
                cloud_std = c.sigma_data
            if "cloud_std" in data:
                cloud_std = data["cloud_std"]
            pos_noised = diffuse_atom_cloud(
                hk.next_rng_key(), pos, mask, batch,
                t_pos[:, None], cloud_std=cloud_std,
                flow=c.diffusion_kind=="flow")
            seq_noised, corrupt_aa = diffuse_sequence(
                hk.next_rng_key(), seq, t_seq)
        pos_noised = pos_noised.to_array()
        # FIXME: center diffused
        # pos_noised = pos_noised - index_mean(pos_noised[:, 1], batch, mask[:, None])[:, None]

        return dict(
            seq_noised=seq_noised, pos_noised=pos_noised,
            t_pos=t_pos, t_seq=t_seq, corrupt_aa=corrupt_aa)

class StructureDiffusionNoise:
    def __init__(self, config):
        self.config = config

    def __call__(self, data):
        c = self.config
        batch = data["batch_index"]
        mask = data["mask"]
        pos = data["pos"]

        pos = Vec3Array.from_array(pos)
        seq = data["seq"]
        t_pos = data["t_pos"]
        t_seq = data["t_seq"]

        if c.diffusion_kind == "edm":
            pos_noised = diffuse_coordinates_edm(
                hk.next_rng_key(), pos, batch, t_pos[:, None])
            seq_noised, corrupt_aa = diffuse_sequence(
                hk.next_rng_key(), seq, t_seq)
        if c.diffusion_kind in ("vp", "vpfixed", "flow"):
            cloud_std = None
            if c.diffusion_kind in ("vpfixed", "flow"):
                cloud_std = c.sigma_data
            if "cloud_std" in data:
                cloud_std = data["cloud_std"]
            pos_noised = diffuse_atom_cloud(
                hk.next_rng_key(), pos, mask, batch,
                t_pos[:, None], cloud_std=cloud_std,
                flow=c.diffusion_kind=="flow")
            seq_noised, corrupt_aa = diffuse_sequence(
                hk.next_rng_key(), seq, t_seq)
        pos_noised = pos_noised.to_array()

        return dict(
            seq_noised=seq_noised, pos_noised=pos_noised,
            t_pos=t_pos, t_seq=t_seq, corrupt_aa=corrupt_aa)

class StructureDiffusionPredict(StructureDiffusionInference):
    """Barebones denoising module to use as a component in a composite diffusion pipeline."""
    def __call__(self, data, prev, fitness):
        c = self.config
        diffusion = Diffusion(c)

        def apply_model(prev):
            result = diffusion(data, prev, fitness, predict_aa=True)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            return result, prev

        if prev is None:
            prev = diffusion.init_prev(data)
        result, prev = apply_model(prev)
        return result, prev

def get_sigma_edm(batch, meanval=None, stdval=None, minval=0.01, maxval=80.0, randomness=None):
    size = batch.shape[0]
    if meanval is not None:
        # if we have specified a log mean & standard deviation
        # for a sigma distribution, sample sigma from that distribution
        if randomness is None:
            randomness = jax.random.normal(hk.next_rng_key(), (size,))[batch]
        log_sigma = meanval + stdval * randomness
        sigma = jnp.exp(log_sigma)
    else:
        # sample from a uniform distribution - this is suboptimal
        # but easier to tune
        if randomness is None:
            randomness = jax.random.uniform(
                hk.next_rng_key(), (size,),
                minval=0, maxval=1)[batch]
        sigma = minval + randomness * (maxval - minval)
    return sigma, randomness

class Encoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "encoder"):
        super().__init__(name)
        self.config = config

    def prepare_features(self, data):
        c = self.config
        positions = data["pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        pair_mask = (batch[:, None] == batch[None, :]) * mask[:, None] * mask[None, :]
        positions = Vec3Array.from_array(positions)
        frames, local_positions = extract_aa_frames(positions)
        augment = VectorLinear(
            c.augment_size,
            initializer=init_linear())(local_positions)
        augment = vector_mean_norm(augment)
        local_positions = jnp.concatenate((
            local_positions.to_array()[:, :5],
            augment.to_array()
        ), axis=-2)
        local_positions = Vec3Array.from_array(local_positions)
        positions = frames[:, None].apply_to_point(local_positions)
        neighbours = get_spatial_neighbours(c.num_neighbours)(
            positions[:, 4], batch, mask)
        _, local_positions = extract_aa_frames(positions)
        dist = local_positions.norm()

        local_features = [
            local_positions.normalized().to_array().reshape(
                local_positions.shape[0], -1),
            distance_rbf(dist, 0.0, 22.0, 16).reshape(
                local_positions.shape[0], -1),
            jnp.log(dist + 1)
        ]
        local = MLP(c.local_size * 4, c.local_size, activation=jax.nn.gelu,
                    bias=False)(
            jnp.concatenate(local_features, axis=-1))
        
        positions = positions.to_array()
        local = hk.LayerNorm([-1], True, True)(local)
        return local, positions, neighbours, resi, chain, batch, mask

    def __call__(self, data):
        c = self.config
        local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data)
        local = EncoderStack(c, c.encoder_depth)(
            local, pos, neighbours, resi, chain, batch, mask)
        # generate augmented vector features from encoder representation
        frames, local_positions = extract_aa_frames(
            Vec3Array.from_array(pos))
        augment = local_positions[:, 5:]
        update = MLP(2 * local.shape[-1], augment.shape[1] * 3,
            activation=jax.nn.gelu, final_init=init_linear())(local)
        update = update.reshape(update.shape[0], augment.shape[1], 3)
        update = Vec3Array.from_array(update)
        augment += update
        local_positions = local_positions[:, :5]
        augment = vector_mean_norm(augment)
        # combine augmented features with backbone / atom positions
        local_positions = jnp.concatenate((
            local_positions.to_array(), augment.to_array()), axis=-2)
        pos = frames[:, None].apply_to_point(
            Vec3Array.from_array(local_positions)).to_array()
        return local, pos

class EncoderBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "encoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos,
                 neighbours, resi, chain, batch, mask):
        c = self.config
        # decide if we're using ResiDual or PreNorm residual branches
        residual_update, residual_input, local_shape = get_residual_gadgets(
            features, c.resi_dual)
        pair_feature_function = encoder_pair_features
        attention = SparseStructureAttention
        # FIXME
        # if c.equivariance == "semi_equivariant":
        #     pair_feature_function = se_encoder_pair_features
        #     attention = SemiEquivariantSparseStructureAttention
        # embed pair features
        pair, pair_mask = pair_feature_function(c)(
                Vec3Array.from_array(pos), neighbours,
                resi, chain, batch, mask)
        # attention / message passing module
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), pos, pair, pair_mask,
                neighbours, resi, chain, batch, mask))
        # global update of local features
        features = residual_update(
            features,
            Update(c, global_update=False)(
                residual_input(features), pos,
                chain, batch, mask, condition=None))
        return features

class EncoderStack(hk.Module):
    def __init__(self, config, depth=None, name: Optional[str] = "self_cond_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 2

    def __call__(self, local, pos, neighbours, resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                data = block(c)(data, pos,
                                neighbours,
                                resi, chain,
                                batch, mask)
                return data
            return _inner
        stack = block_stack(
            self.depth, block_size=1, with_state=False)(
                hk.remat(stack_inner(EncoderBlock)))
        if c.resi_dual:
            incremental = local
            local, incremental = stack((local, incremental))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local = hk.LayerNorm([-1], True, True)(stack(local))
        return local

class AADecoderBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "aa_decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, features, pos,
                 neighbours, resi, chain, batch, mask,
                 priority=None):
        c = self.config
        # decide if we're using ResiDual or PreNorm residual branches
        residual_update, residual_input, _ = get_residual_gadgets(features, c.resi_dual)
        # embed pair features
        pair, pair_mask = aa_decoder_pair_features(c)(
            Vec3Array.from_array(pos), aa, neighbours, resi, chain, batch, mask)
        if priority is not None:
            pair_mask *= priority[:, None] > priority[neighbours]
        # attention / message passing module
        features = residual_update(
            features,
            SparseStructureAttention(c)(
                residual_input(features), pos, pair, pair_mask,
                neighbours, resi, chain, batch, mask))
        # local feature transition (always enabled)
        features = residual_update(
            features,
            EncoderUpdate(c)(residual_input(features), pos, chain, batch, mask))
        return features

class AADecoderStack(hk.Module):
    def __init__(self, config, depth=None, name: Optional[str] = "aa_decoder_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 3

    def __call__(self, aa, local, pos, neighbours,
                 resi, chain, batch, mask, priority=None):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                data = block(c)(aa, data, pos,
                                neighbours,
                                resi, chain,
                                batch, mask,
                                priority=priority)
                return data
            return _inner
        stack = block_stack(
            self.depth, block_size=1, with_state=False)(
                hk.remat(stack_inner(AADecoderBlock)))
        if c.resi_dual:
            incremental = local
            local, incremental = stack((local, incremental))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local = hk.LayerNorm([-1], True, True)(stack(local))
        return local

def edm_scaling(sigma_data, sigma_noise, alpha=1.0, beta=1.0):
    sigma_data = jnp.maximum(sigma_data, 1e-3)
    sigma_noise = jnp.maximum(sigma_noise, 1e-3)
    denominator = alpha ** 2 * sigma_data ** 2 + beta ** 2 * sigma_noise ** 2
    in_scale = 1 / jnp.sqrt(denominator)
    refine_in_scale = 1 / sigma_data
    out_scale = beta * sigma_data * sigma_noise / jnp.sqrt(denominator)
    skip_scale = alpha * sigma_data ** 2 / denominator
    loss_scale = 1 / out_scale ** 2
    return {"in": in_scale, "out": out_scale, "skip": skip_scale,
            "refine": refine_in_scale, "loss": loss_scale}

def scale_to_center(positions, sigma, skip_scale,
                    chain_index, mask):
    if not isinstance(skip_scale, float):
        skip_scale = skip_scale[..., None]
    center = index_mean(
        positions[:, 1], chain_index,
        mask[..., None], weight=sigma)
    positions -= center[:, None, :]
    positions *= skip_scale
    positions += center[:, None, :]
    return positions

def preconditioning_scale_factors(config, t, sigma_data):
    c = config
    if c.preconditioning in ("edm", "edm_scaled"):
        result = edm_scaling(sigma_data, 1.0, alpha=1.0, beta=t)
    elif c.preconditioning == "flow":
        result = {"in": 1.0, "out": sigma_data, "skip": 1.0,
                  "refine": 1, "loss": 1 / jnp.maximum(t, 0.01) ** 2}
    else:
        result = {"in": 1.0, "out": sigma_data, "skip": 1.0,
                  "refine": 1, "loss": 1}
    if not c.refine:
        result["refine"] = 1.0
    return result

class Diffusion(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config
        self.diffusion_block = GradientDiffusionBlock

    def __call__(self, data, prev, fitness, override=None, predict_aa=False):
        c = self.config
        self.diffusion_block = GradientDiffusionBlock
        diffusion_stack_module = DiffusionStack
        if c.preconditioning == "edm_scaled":
            diffusion_stack_module = EDMDiffusionStack
        diffusion_stack = diffusion_stack_module(c, self.diffusion_block)
        distogram_decoder = DistogramDecoder(c)
        features = self.prepare_features(data, prev, override=override)
        local, pos, prev_pos, t_pos, t_seq, resi, chain, batch, mask = features

        # get neighbours for distogram supervision
        pos_gt = Vec3Array.from_array(data["pos"])
        distogram_neighbours = get_random_neighbours(c.fape_neighbours)(pos_gt[:, 4], batch, mask)
        sup_neighbours = distogram_neighbours
        if not c.distogram_trajectory:
            sup_neighbours = None
        
        # enable selective cyclisation of chains
        cyclic_mask = None
        if c.cyclic and "cyclic_mask" in data:
            cyclic_mask = data["cyclic_mask"]

        pos_factors = preconditioning_scale_factors(c, t_pos, c.sigma_data)
        skip_scale_pos = pos_factors["skip"]
        if c.preconditioning == "edm":
            pos *= skip_scale_pos[:, None, None]
        if c.preconditioning == "edm_scaled":
            local, pos, trajectory, dssp, aa = diffusion_stack(
                local, pos, prev_pos,
                fitness,
                t_pos, resi, chain, batch, mask,
                cyclic_mask=cyclic_mask,
                sup_neighbours=sup_neighbours)
        else:
            local, pos, trajectory, dssp, aa = diffusion_stack(
                local, pos, prev_pos,
                fitness,
                t_pos, resi, chain, batch, mask,
                cyclic_mask=cyclic_mask,
                sup_neighbours=sup_neighbours)
        # predict & handle sequence, losses etc.
        result = dict()
        if c.distogram_trajectory:
            trajectory, distogram_trajectory = trajectory
            result["distogram_trajectory"] = distogram_trajectory
        result["aa_trajectory"] = aa
        result["dssp_trajectory"] = dssp
        result["trajectory"] = trajectory
        result["local"] = local
        result["pos"] = pos
        result["aa"] = aa[-1]
        result["dssp"] = dssp[-1]
        result["aa_gt"] = data["aa_gt"]
        result["pos_noised"] = data["pos_noised"]
        # distogram decoder features
        result["distogram_neighbours"] = distogram_neighbours
        result["distogram"] = distogram_decoder(local, pos, distogram_neighbours,
                                                resi, chain, batch, mask,
                                                cyclic_mask=cyclic_mask)
        # generate all-atom positions using predicted
        # side-chain torsion angles:
        aatype = data["aa_gt"]
        if predict_aa:
            aatype = jnp.argmax(aa[-1], axis=-1)
        result["aatype"] = aatype
        raw_angles, angles, angle_pos = get_angle_positions(
            aatype, local, pos)
        result["raw_angles"] = raw_angles
        result["angles"] = angles
        result["atom_pos"] = angle_pos
        if c.encode_atom14:
            result["atom_pos"] = pos
        return result

    def init_prev(self, data):
        c = self.config
        return {
            "pos": 0.0 * jax.random.normal(hk.next_rng_key(), data["pos_noised"].shape),
            "local": jnp.zeros((data["pos_noised"].shape[0], c.local_size), dtype=jnp.float32)
        }

    def prepare_features(self, data, prev, override=None):
        c = self.config
        pos = data["pos_noised"]
        t_pos = data["t_pos"]   
        seq = data["seq_noised"]
        t_seq = data["t_seq"]
        if override is not None:
            pos = override["pos_noised"]
            t_pos = override["t_pos"]
            seq = override["seq_noised"]
            t_seq = override["t_seq"]
        if c.mask_atom14:
            pos = pos.at[:, 5:].set(pos[:, 4:5])
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        edm_time_embedding = lambda x: jnp.concatenate((
            jnp.log(x[:, None]) / 4,
            fourier_time_embedding(jnp.log(x) / 4, size=256)
        ), axis=-1)
        vp_time_embedding = lambda x: jnp.concatenate((
            x[:, None],
            fourier_time_embedding(x, size=256)
        ), axis=-1)
        if c.diffusion_kind in ("vp", "vpfixed", "chroma", "flow"):
            time_embedding = vp_time_embedding
        else:
            time_embedding = edm_time_embedding
        pos_time_features = time_embedding(t_pos)
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1)
        ]
        if c.time_embedding:
            local_features = [pos_time_features] + local_features
        if c.diffuse_sequence:
            seq_time_features = time_embedding(t_seq)
            local_features += [
                seq_time_features,
                jax.nn.one_hot(seq, 21, axis=-1)
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

        return local, pos, prev["pos"], t_pos, t_seq, resi, chain, batch, mask

    def loss(self, data, result):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        losses = dict()
        total = 0.0

        # mask for losses which should only apply late
        # in the diffusion process
        if c.diffusion_kind == "edm":
            late_mask = data["t_pos"] < 5.0
        else:
            late_mask = data["t_pos"] < 0.5
        late_mask *= mask
        
        # AA NLL loss
        aa_predict_mask = mask * (data["aa_gt"] != 20)
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll
        total += c.aa_weight * aa_nll

        # AA trajectory loss
        if "aa_trajectory" in result:
            aa_predict_mask = mask * (data["aa_gt"] != 20)
            aa_trajectory_nll = -(result["aa_trajectory"] * jax.nn.one_hot(data["aa_gt"][None], 20, axis=-1)).sum(axis=-1)
            aa_trajectory_nll = jnp.where(aa_predict_mask[None], aa_trajectory_nll, 0)
            aa_trajectory_nll = aa_trajectory_nll.sum(axis=-1) / jnp.maximum(aa_predict_mask.sum()[None], 1)
            aa_trajectory_nll *= 0.99 ** jnp.arange(aa_trajectory_nll.shape[0])[::-1]
            aa_trajectory_nll = aa_trajectory_nll.mean()
            losses["aa_trajectory"] = aa_trajectory_nll
            total += c.aa_weight * aa_trajectory_nll

        # DSSP NLL loss
        dssp_nll = -(result["dssp"] * jax.nn.one_hot(data["dssp_gt"], 3, axis=-1)).sum(axis=-1)
        dssp_nll = jnp.where(mask, dssp_nll, 0)
        dssp_nll = dssp_nll.sum() / jnp.maximum(mask.sum(), 1)
        losses["dssp"] = dssp_nll
        total += 1.0 * dssp_nll

        # DSSP trajectory loss
        if "dssp_trajectory" in result:
            dssp_trajectory_nll = -(result["dssp_trajectory"] * jax.nn.one_hot(data["dssp_gt"][None], 3, axis=-1)).sum(axis=-1)
            dssp_trajectory_nll = jnp.where(mask[None], dssp_trajectory_nll, 0)
            dssp_trajectory_nll = dssp_trajectory_nll.sum(axis=-1) / jnp.maximum(mask.sum()[None], 1)
            dssp_trajectory_nll *= 0.99 ** jnp.arange(dssp_trajectory_nll.shape[0])[::-1]
            dssp_trajectory_nll = dssp_trajectory_nll.mean()
            losses["dssp_trajectory"] = dssp_trajectory_nll
            total += dssp_trajectory_nll

        # diffusion losses
        base_weight = mask / jnp.maximum(index_sum(mask.astype(jnp.float32), batch, mask), 1) / (batch.max() + 1)
        # diffusion (z-space)
        do_clip = jnp.where(
            jax.random.bernoulli(hk.next_rng_key(), c.p_clip, batch.shape)[batch],
            100.0,
            jnp.inf)
        sigma = data["t_pos"]
        loss_weight = preconditioning_scale_factors(
            c, sigma, c.sigma_data)["loss"]
        diffusion_weight = base_weight * loss_weight
        pos_gt = data["pos"]
        pos_pred = result["pos"]
        if c.kabsch:
            pos_gt = index_align(pos_gt, pos_pred, batch, mask)
        dist2 = ((result["pos"] - pos_gt) ** 2).sum(axis=-1)
        dist2 = jnp.clip(dist2, 0, do_clip[:, None]).mean(axis=-1)
        dist2 = jnp.where(mask, dist2, 0)
        pos_loss = (dist2 * diffusion_weight).sum() / 3
        losses["pos"] = pos_loss
        total += c.pos_weight * pos_loss
        # diffusion (z-space trajectory)
        dist2 = ((result["trajectory"] - pos_gt[None]) ** 2).sum(axis=-1)
        dist2 = jnp.clip(dist2, 0, do_clip[None, :, None]).mean(axis=-1)
        dist2 = jnp.where(mask[None], dist2, 0)
        trajectory_pos_loss = (dist2 * diffusion_weight[None, ...]).sum(axis=1) / 3
        trajectory_pos_loss = trajectory_pos_loss.mean()
        losses["pos_trajectory"] = trajectory_pos_loss    
        total += c.trajectory_weight * trajectory_pos_loss
        # diffusion (x-space)
        atom_pos_pred = result["atom_pos"]
        atom_pos_gt = data["atom_pos"]
        if c.kabsch:
            atom_pos_gt = index_align(atom_pos_gt, atom_pos_pred, batch, mask)
        dist2 = ((atom_pos_gt - result["atom_pos"]) ** 2).sum(axis=-1)
        dist2 = jnp.clip(dist2, 0, do_clip[:, None])
        atom_mask = data["atom_mask"]
        if c.x_late:
            atom_mask *= late_mask[..., None]
        x_loss = (jnp.where(atom_mask, dist2, 0)).sum(axis=-1) / 3
        x_loss /= jnp.maximum(atom_mask.sum(axis=-1), 1)
        x_loss = (x_loss * base_weight).sum()
        losses["x"] = x_loss
        total += c.x_weight * x_loss
        # diffusion (rotation-space)
        gt_frames, _ = extract_aa_frames(
            Vec3Array.from_array(data["atom_pos"]))
        frames, _ = extract_aa_frames(
            Vec3Array.from_array(result["atom_pos"]))
        rotation_product = (gt_frames.rotation.inverse() @ frames.rotation).to_array()
        rotation_loss = ((rotation_product - jnp.eye(rotation_product.shape[-1])) ** 2).sum(axis=(-1, -2))
        rotation_loss = jnp.where(mask, rotation_loss, 0)
        rotation_loss = (rotation_loss * base_weight).sum()
        losses["rotation"] = rotation_loss
        total += c.rotation_weight * rotation_loss
        frames, _ = jax.vmap(extract_aa_frames)(
            Vec3Array.from_array(result["trajectory"]))
        rotation_product = (gt_frames.rotation.inverse()[None] @ frames.rotation).to_array()
        rotation_loss = ((rotation_product - jnp.eye(rotation_product.shape[-1])) ** 2).sum(axis=(-1, -2))
        rotation_loss = jnp.where(mask[None], rotation_loss, 0)
        rotation_loss = (rotation_loss * base_weight[None]).sum(axis=-1)
        rotation_loss = rotation_loss.mean()
        losses["rotation_trajectory"] = rotation_loss
        total += c.rotation_trajectory_weight * rotation_loss

        # sparse neighbour FAPE ** 2
        pair_mask = batch[:, None] == batch[None, :]
        pair_mask *= mask[:, None] * mask[None, :]
        pos_gt = data["pos"]
        pos_gt = jnp.where(mask[:, None, None], pos_gt, 0)
        pos_gt = Vec3Array.from_array(pos_gt)
        # FIXME: do we have to stop gradients through the encoder? is this what fucks us?
        frames_gt, _ = extract_aa_frames(jax.lax.stop_gradient(pos_gt))
        # CB distance
        distance = (pos_gt[:, None, 4] - pos_gt[None, :, 4]).norm()
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

        # local FAPE
        atom_mask = data["atom_mask"]
        local_neighbours = get_neighbours(c.local_neighbours)(distance, pair_mask)
        gt_pos_neighbours = Vec3Array.from_array(
            data["atom_pos"][local_neighbours])
        pos_neighbours = Vec3Array.from_array(
            result["atom_pos"][local_neighbours])
        mask_neighbours = (local_neighbours != -1)[..., None] * data["atom_mask"][local_neighbours]
        gt_local_positions = gt_frames[:, None, None].apply_inverse_to_point(
            gt_pos_neighbours)
        local_positions = frames[-1, :, None, None].apply_inverse_to_point(
            pos_neighbours)
        dist = (local_positions - gt_local_positions).norm()
        dist2 = jnp.clip(dist, 0, 10.0)
        local_loss = (jnp.where(mask_neighbours, dist2, 0)).sum(axis=(-1, -2)) / 3
        local_loss /= jnp.maximum(mask_neighbours.astype(jnp.float32).sum(axis=(-1, -2)), 1)
        local_loss = (local_loss * base_weight * late_mask).sum()
        losses["local"] = local_loss
        total += c.local_weight * local_loss

        # sparse distogram loss
        neighbours = result["distogram_neighbours"]
        pair_mask = (neighbours != -1) * mask[:, None] * mask[neighbours]
        cb = pos_gt[:, 4]
        distance_gt = (cb[:, None] - cb[neighbours]).norm()
        distogram_gt = jax.lax.stop_gradient(distance_one_hot(distance_gt))
        distogram_nll = -(result["distogram"] * distogram_gt).sum(axis=-1)
        distogram_nll = jnp.where(pair_mask, distogram_nll, 0).sum(axis=-1) / jnp.maximum(pair_mask.sum(axis=-1), 1)
        distogram_nll = (distogram_nll * base_weight).sum()
        losses["distogram"] = distogram_nll
        total += 0.1 * distogram_nll

        # all-atom soft LDDT
        if c.soft_lddt:
            pos_gt = Vec3Array.from_array(data["pos"])
            pos_pred = Vec3Array.from_array(result["trajectory"][-1])
            dist_gt = (pos_gt[:, None, :, None] - pos_gt[neighbours, None, :]).norm()
            dist_pred = (pos_pred[:, None, :, None] - pos_pred[neighbours, None, :]).norm()
            dd = abs(dist_gt - dist_pred)
            dd_mask = pair_mask[..., None, None] * (dist_gt < 15.0)
            dd_mask = dd_mask * (1 - jnp.eye(pos_gt.shape[-1]))[None, None] > 0
            ddt = jax.nn.sigmoid(0.5 - dd) + jax.nn.sigmoid(1 - dd) + jax.nn.sigmoid(2 - dd) + jax.nn.sigmoid(4 - dd)
            atom_lddt = ((ddt / 4) * dd_mask).sum(axis=(-1, -3)) / jnp.maximum(dd_mask.sum(axis=(-1, -3)), 1)
            residue_lddt = atom_lddt.mean(axis=-1)
            mean_lddt = (base_weight * residue_lddt).sum()
            losses["lddt"] = mean_lddt
            total += c.local_weight * (1 - mean_lddt)

        # distogram trajectory loss?
        if c.distogram_trajectory:
            distogram_gt = jax.lax.stop_gradient(distance_one_hot(distance_gt, bins=16))
            distogram_nll = -(result["distogram_trajectory"] * distogram_gt[None]).sum(axis=-1)
            distogram_nll = jnp.where(pair_mask[None], distogram_nll, 0).sum(axis=-1) / jnp.maximum(pair_mask[None].sum(axis=-1), 1)
            distogram_nll = (distogram_nll * base_weight[None]).sum(axis=1)
            distogram_nll = distogram_nll.mean()
            losses["distogram_trajectory"] = distogram_nll
            total += 0.1 * distogram_nll

        # violation loss
        res_mask = data["mask"] * late_mask
        pred_mask = get_atom14_mask(data["aa_gt"]) * res_mask[:, None]
        violation, _ = violation_loss(data["aa_gt"],
                                    data["residue_index"],
                                    result["atom_pos"],
                                    pred_mask,
                                    res_mask,
                                    clash_overlap_tolerance=1.5,
                                    violation_tolerance_factor=2.0,
                                    chain_index=data["chain_index"],
                                    batch_index=data["batch_index"],
                                    per_residue=False)
        losses["violation"] = violation.mean()
        total += c.violation_weight * violation.mean()
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

class DistogramDecoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "distogram_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, neighbours, resi, chain, batch, mask, cyclic_mask=None):
        c = self.config
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pos = Vec3Array.from_array(pos)
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, cyclic=cyclic_mask is not None)(
                resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
        if not c.distogram_no_pos:
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                direction_features(pos, neighbours))
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                position_rotation_features(pos, neighbours))
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                pair_vector_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(local)[:, None]
        pair += Linear(c.pair_size, bias=False, initializer="linear")(local)[neighbours]
        pair = hk.LayerNorm([-1], True, True)(pair)
        distogram = MLP(
            pair.shape[-1] * 2, 64,
            activation=jax.nn.gelu,
            final_init="zeros")(pair)
        return jax.nn.log_softmax(distogram, axis=-1)

class AADecoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "aa_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, local, pos, resi, chain, batch, mask,
                 priority=None):
        c = self.config
        if c.aa_decoder_kind == "adm":
            local += Linear(
                local.shape[-1],
                initializer=init_glorot())(jax.nn.one_hot(aa, 21))
        neighbours = extract_neighbours(num_index=0, num_spatial=32, num_random=0)(
            Vec3Array.from_array(pos), resi, chain, batch, mask)
        if not c.linear_aa:
            local = AADecoderStack(c, depth=c.aa_decoder_depth)(
                aa, local, pos, neighbours,
                resi, chain, batch, mask,
                priority=priority)
            local = hk.LayerNorm([-1], True, True)(local)
        return Linear(20, initializer=init_zeros(), bias=False)(local), local

    def train(self, aa, local, pos, resi, chain, batch, mask):
        c = self.config
        if c.aa_decoder_kind == "adm":
            # if we're training the decoder to be an autoregressive
            # diffusion model:
            # sample a random time t between 0 and 1
            t = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
            # mask out 100 * t % of the input sequence
            # and return `corrupt_aa` as an indicator
            # which amino acids were masked
            aa, corrupt_aa = diffuse_sequence(hk.next_rng_key(), aa, t)
            # compute amino acid logits
            logits, features = self(
                aa, local, pos, resi, chain, batch, mask)
            logits = jax.nn.log_softmax(logits, axis=-1)
            return logits, features, corrupt_aa
        if c.aa_decoder_kind == "ar":
            # if we're training a random-order autoregressive
            # decoder (a la Protein MPNN):
            # sample an array of priorities for each amino acid
            # without replacement. Lower priority amino acids
            # will be sampled before higher-priority amino acids.
            priority = jax.random.permutation(
                hk.next_rng_key(), aa.shape[0])
            # compute amino acid logits
            logits, features = self(
                aa, local, pos, resi, chain, batch, mask,
                priority=priority)
            logits = jax.nn.log_softmax(logits)
            return logits, features, jnp.ones_like(mask)
        # otherwise, mask out all positions and
        # independently predict logits for each
        # sequence position
        aa = 20 * jnp.ones_like(aa)
        logits, features = self(
            aa, local, pos, resi, chain, batch, mask)
        logits = jax.nn.log_softmax(logits)
        return logits, features, jnp.ones_like(mask)

class DiffusionStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, prev_pos,
                 fitness, time,
                 resi, chain, batch, mask,
                 cyclic_mask=None,
                 sup_neighbours=None):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the diffusion block
                result = block(c)(
                    features, pos, prev_pos,
                    fitness, time,
                    resi, chain, batch, mask,
                    cyclic_mask=cyclic_mask)
                features, pos, dssp, aa = result
                trajectory_output = (pos, dssp, aa)
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), trajectory_output
            return _inner
        diffusion_block = self.block
        stack = block_stack(
            c.diffusion_depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(diffusion_block)))
        if c.resi_dual:
            # handle ResiDual local features
            incremental = local
            ((local, incremental), pos), trajectory = stack(
                ((local, incremental), pos))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            (local, pos), trajectory = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        trajectory, dssp, aa = trajectory
        return local, pos, trajectory, dssp, aa
    
class EDMDiffusionStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, prev_pos,
                 fitness, time,
                 resi, chain, batch, mask,
                 cyclic_mask=None,
                 sup_neighbours=None):
        c = self.config
        scale = edm_scaling(c.sigma_data, 1.0, alpha=1.0, beta=time)
        init_pos = c.sigma_data * scale["in"][:, None, None] * pos
        pos = pos * scale["skip"][:, None, None]
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the diffusion block
                result = block(c)(
                    features, pos, prev_pos,
                    fitness, time,
                    resi, chain, batch, mask,
                    cyclic_mask=cyclic_mask,
                    init_pos=init_pos)
                features, pos, dssp, aa = result
                trajectory_output = (pos, dssp, aa)
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), trajectory_output
            return _inner
        diffusion_block = self.block
        stack = block_stack(
            c.diffusion_depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(diffusion_block)))
        if c.resi_dual:
            # handle ResiDual local features
            incremental = local
            ((local, incremental), pos), trajectory = stack(
                ((local, incremental), pos))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            (local, pos), trajectory = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        trajectory, dssp, aa = trajectory
        return local, pos, trajectory, dssp, aa

def encoder_pair_features(c):
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
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

def aa_decoder_pair_features(c):
    def inner(pos, aa, neighbours, resi, chain, batch, mask):
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
        pair += hk.LayerNorm([-1], True, True)(
            Linear(pair.shape[-1], bias=False)(
                jax.nn.one_hot(aa, 21)))[neighbours]
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

def se_encoder_pair_features(c):
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        local_neighbourhood = jnp.concatenate((
            jnp.repeat(pos.to_array()[:, None], neighbours.shape[1], axis=1),
            pos.to_array()[neighbours]
        ), axis=-2) / c.sigma_data
        local_neighbourhood = local_neighbourhood.reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            local_neighbourhood)
        vectors = VectorLinear(5, initializer="linear")(pos - pos[:, 1, None]) + pos[:, 1, None]
        dirs: jnp.ndarray = (vectors[:, None, :, None] - vectors[neighbours, None, :]).to_array() / c.sigma_data
        dirs = dirs.reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(dirs)
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

def dmap_pair_features(c):
    def inner(pos, dmap, neighbours,
              resi, chain, batch, mask,
              cyclic_mask=None):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        pair = Linear(c.pair_size, bias=False)(
            distance_rbf(dmap[index[:, None], neighbours], 0.0, 22.0, 16.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(Vec3Array.from_array(pos),
                              neighbours, d_min=0.0, d_max=22.0))
        pair = hk.LayerNorm([-1], True, True)(pair)
        return MLP(pair.shape[-1], pair.shape[-1], final_init="linear")(pair), pair_mask
    return inner

def dmap_local_update(c):
    pair_features = dmap_pair_features(c)
    def inner(features, pos, dmap, neighbours,
              resi, chain, batch, mask):
        pair, pair_mask = pair_features(
            pos, dmap, neighbours,
            resi, chain, batch, mask,
            cyclic_mask=None)
        message = jnp.where(pair_mask[..., None], pair, 0).sum(axis=1)
        message /= jnp.maximum(pair_mask[..., None].sum(axis=1), 1)
        weight = jax.nn.gelu(Linear(message.shape[-1], bias=False)(features))
        message *= weight
        return Linear(features.shape[-1], bias=False, initializer="zeros")(message)
    return inner

def gradient_diffusion_pair_features(c):
    def inner(pos, neighbours,
              resi, chain, batch, mask, cyclic_mask=None, add_pos=None):
        if add_pos is None:
            add_pos = []
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, cyclic=cyclic_mask is not None)(
                resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
        pos = Vec3Array.from_array(pos)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        for pos in add_pos:
            pos = Vec3Array.from_array(pos)
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                position_rotation_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        return pair, pair_mask
    return inner

class GradientUpdate(hk.Module):
    def __init__(self, config, name: str | None = "gradient_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, fitness, features, pos):
        c = self.config
        dssp_pred = jax.nn.log_softmax(MLP(
            features.shape[-1] * 2, 3, bias=False,
            activation=jax.nn.gelu, final_init="zeros")(features), axis=-1)
        aa_pred = jax.nn.log_softmax(MLP(
            features.shape[-1] * 2, 20, bias=False,
            activation=jax.nn.gelu, final_init="zeros")(features), axis=-1)
        # TODO: check correctness in training fitness
        pos = pos[:, 1]
        # add pos noise to induce robustness to inaccurate gradients
        if not c.eval:
            pos += 0.01 * jax.random.normal(hk.next_rng_key(), pos.shape)
        pos_grad, aa_grad, dssp_grad = hk.grad(
            fitness, argnums=(0, 1, 2))(
                pos, aa_pred, dssp_pred)
        if c.eval:
            norm_dssp = jnp.linalg.norm(dssp_grad, axis=-1).max()
            norm_pos = jnp.linalg.norm(pos_grad, axis=-1).max()
            jax.debug.print(
                "grad norms: dssp {norm_dssp} pos {norm_pos}",
                norm_dssp=norm_dssp, norm_pos=norm_pos)
        frames, _ = extract_aa_frames(Vec3Array.from_array(pos))
        pos_grad = frames.rotation.apply_inverse_to_point(
            Vec3Array.from_array(pos_grad)).to_array()
        pos_grad /= c.sigma_data
        grad_update = MLP(
            features.shape[-1] * 4, features.shape[-1],
            activation=jax.nn.gelu, bias=False, final_init="zeros")(
                jnp.concatenate((dssp_grad, aa_grad, pos_grad), axis=-1)
            )
        return grad_update, pos_grad, dssp_pred, aa_pred

class GradientDiffusionBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, prev_pos,
                 fitness, time,
                 resi, chain, batch, mask,
                 cyclic_mask=None, init_pos=None):
        c = self.config
        residual_update, residual_input, _ = get_residual_gadgets(
            features, c.resi_dual)
        neighbours = extract_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=32)(
                Vec3Array.from_array(pos),
                resi, chain, batch, mask)
        grad_update, pos_grad, dssp_pred, aa_pred = GradientUpdate(c)(
            fitness, residual_input(features), pos)
        features = residual_update(features, grad_update)

        # TODO: remove mention of "condition" and "pair_condition"
        pair_feature_function = gradient_diffusion_pair_features
        attention = SparseStructureAttention
        add_pos = [prev_pos]
        if init_pos is not None:
            add_pos = [init_pos] + add_pos
        pair, pair_mask = pair_feature_function(c)(
            pos, neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask, add_pos=add_pos)
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), pos / c.sigma_data, pair, pair_mask,
                neighbours, resi, chain, batch, mask))
        # global update of local features
        features = residual_update(
            features,
            Update(c, global_update=True)(
                residual_input(features),
                pos, chain, batch, mask))
        if c.resi_dual:
            local, incremental = features
            local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local_norm = residual_input(features)
        # NOTE: this is sigma_data for VP and flow,
        # but the proper output scaling for VE/EDM
        out_scale = preconditioning_scale_factors(
            c, time[:, None], c.sigma_data)["out"]
        position_update_function = update_positions
        pos = position_update_function(
            pos, local_norm,
            scale=out_scale,
            symm=c.symm)
        return features, pos.astype(local_norm.dtype), dssp_pred, aa_pred

def get_residual_gadgets(features, use_resi_dual=True):
    residual_update = (lambda x, y: resi_dual(*x, y)) if use_resi_dual else prenorm_skip
    residual_input = resi_dual_input if use_resi_dual else prenorm_input
    local_shape = features[0].shape[-1] if use_resi_dual else features.shape[-1]
    return residual_update, residual_input, local_shape

def make_pair_mask(mask, neighbours):
    return mask[:, None] * mask[neighbours] * (neighbours != -1)

class Update(hk.Module):
    def __init__(self, config, global_update=True,
                 name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config
        self.global_update = global_update

    def __call__(self, local, pos, chain, batch, mask, condition=None):
        c = self.config
        if c.equivariance == "semi_equivariant":
            local_pos = (pos - pos[:, None, 1]).reshape(pos.shape[0], -1)
        else:
            _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
            local_pos = local_pos.to_array().reshape(local_pos.shape[0], -1)
        if condition is not None:
            local += Linear(local.shape[-1], initializer=init_zeros(), bias=False)(condition)
        local += MLP(local.shape[-1] * 2,
                     local.shape[-1],
                     activation=jax.nn.gelu,
                     final_init=init_zeros())(local_pos)
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        if self.global_update:
            chain_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
            batch_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        hidden = local_gate * local_update
        if self.global_update:
            hidden += index_mean(batch_gate * local_update, batch, mask[..., None])
            hidden += index_mean(chain_gate * local_update, chain, mask[..., None])
        result = Linear(local.shape[-1], initializer=init_zeros())(hidden)
        return result

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

class DiffusionUpdate(hk.Module):
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, condition, chain, batch, mask):
        c = self.config
        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        local_pos = local_pos.to_array().reshape(local_pos.shape[0], -1)
        local += Linear(local.shape[-1], initializer=init_zeros(), bias=False)(condition)
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

def update_positions_precomputed(pos, pos_update, scale=10.0, symm=None):
    frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
    pos_update *= scale
    if symm is not None:
        pos_update = symm(pos_update)
    pos_update = Vec3Array.from_array(pos_update)
    local_pos += pos_update

    # project updated pos to global coordinates
    pos = frames[..., None].apply_to_point(local_pos).to_array()
    return pos

POLAR_THRESHOLD = 3.0
CONTACT_THRESHOLD = 8.0
def interactions(coords, block_adjacency, mask, resi, chain, batch, block_noise=None):
    coords = coords[:, 1]
    same_batch = batch[:, None] == batch[None, :]
    same_chain = (chain[:, None] == chain[None, :]) * same_batch
    mask = mask[:, 1]
    pair_mask = mask[:, None] * mask[None, :]
    pair_mask *= same_batch
    # donor_mask = make_atom14_donor_mask(aatype) # TODO
    # acceptor_mask = make_atom14_acceptor_mask(aatype) # TODO
    # polar_mask = acceptor_mask[:, None, :, None] * donor_mask[None, :, None, :] + donor_mask[:, None, :, None] * acceptor_mask[None, :, None, :]
    distances = jnp.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    distances = jnp.where(pair_mask, distances, jnp.inf)
    # polar_interaction = jnp.where(polar_mask, distances, jnp.inf).min(axis=(-1, -2)) <= POLAR_THRESHOLD
    contact_interaction = distances <= CONTACT_THRESHOLD

    size = resi.shape[0]
    # which chains are in contact with each other at all?
    chain_contact = jnp.zeros((size // 10, size // 10), dtype=jnp.bool_).at[chain[:, None], chain[None, :]].max(contact_interaction)
    chain_contact = jnp.where(jnp.eye(size // 10, size // 10), 0, chain_contact)
    chain_contact = chain_contact[chain[:, None], chain[None, :]]
    chain_contact = jnp.where(same_batch, chain_contact, 0)
    chain_contact = jnp.where(same_chain, 0, chain_contact)
    # chain_contact = jnp.zeros_like(same_chain)
    # which residues on this chain interact with anything else?
    hotspot = jnp.where(same_chain, 0, contact_interaction).any(axis=1)
    # which residues on another chain are interaction hotspots as seen by this chain?
    # take the union of all contacts for each chain across all positions in all other chains.
    relative_hotspot = jnp.zeros((size // 10, contact_interaction.shape[1]), dtype=jnp.bool_).at[chain, :].max(
        jnp.where(same_chain, 0, contact_interaction))[chain]
    # the transpose of this tells us which hotspot residues on this chain interact with another chain
    # we should include both in attention computations, as any kind of hotspot specification should
    # be seen by both chains in the granularity it is provided.
    # E.g. if we specify 5 hotspots on chain A that should be bound by any residue on chain B,
    # then chain A should know that these 5 hotspots are going to be bound by at least one residue on chain B
    # and all residues on chain B should know that at least one of them should bind anywhere on chain A

    block_contact = block_adjacency
    return dict(
        hotspot=hotspot,
        chain_contact=chain_contact,
        relative_hotspot=relative_hotspot,
        block_contact=block_contact,
    )
