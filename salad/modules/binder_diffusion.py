from typing import Optional
from functools import partial

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
    index_max, index_mean, index_sum, index_count, extract_aa_frames,
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
    diffuse_coordinates_edm, diffuse_target_centered,
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

class BinderDiffusion(hk.Module):
    def __init__(self, config,
                 name: Optional[str] = "binder_diffusion"):
        super().__init__(name)
        self.config = config

    def prepare_data(self, data):
        c = self.config
        pos = data["all_atom_positions"]
        atom_mask = data["all_atom_mask"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        is_target = data["is_target"]

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

        # compute hotspots for conditioning
        # get CB distances of binder target complex
        cb = pos[:, 4]
        dist = jnp.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
        # invalidate all target-target and binder-(X) distances
        # leaving only target-binder distances
        dist = jnp.where(is_target[:, None] * (~is_target[None, :]), dist, jnp.inf)
        # find contacts or near-contacts (dist-CB < 8)
        dist = dist.min(axis=1)
        hotspots = dist < 8
        # subsample hotspots
        hotspot_weight = hotspots * jax.random.uniform(hk.next_rng_key(), hotspots.shape)
        hotspot_max = index_max(hotspot_weight, batch, mask)
        center_hotspot = hotspot_weight == hotspot_max
        hotspots = hotspot_weight >= 0.5 * hotspot_max

        # if specified, augment backbone with encoded positions
        data["chain_index"] = chain
        data["hotspots"] = hotspots
        data["center_hotspot"] = center_hotspot
        data["pos"] = pos
        data["atom_mask"] = atom_mask
        data["mask"] = mask
        target, seq, pos = self.encode(data)
        target = jnp.where(is_target[:, None], target, 0)

        # set all-atom-position target
        atom_pos = pos_14
        atom_mask = atom_mask_14
        return dict(seq=seq, pos=pos, target=target,
                    chain_index=chain, mask=mask,
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
            target, augmented_pos = Encoder(c)(data)
            pos = augmented_pos
        else:
            pos = backbone
        seq = aatype
        return target, seq, pos

    def prepare_condition(self, data):
        # only use DSSP condition for binder design
        c = self.config
        aa = data["aa_gt"]
        pos = data["atom_pos"]
        pos_mask = data["atom_mask"]
        dssp, blocks, block_adjacency = assign_dssp(pos, data["batch_index"], pos_mask.any(axis=-1))
        _, block_adjacency = drop_dssp(
            hk.next_rng_key(), dssp, blocks, block_adjacency, p_drop=c.p_drop_dssp)
        condition, aa_mask, dssp_mask = Condition(c)(
            aa, dssp, pos_mask.any(axis=-1),
            data["residue_index"], data["chain_index"],
            data["batch_index"], set_condition=None)

        # include conditioning stochastically during training
        use_condition = jax.random.bernoulli(hk.next_rng_key(), 0.5, (aa.shape[0],))
        use_condition = use_condition[data["batch_index"]]

        condition = jnp.where(use_condition[..., None], condition, 0)
        result = dict(condition=condition,
                      dssp_gt=dssp,
                      aa_mask=jnp.where(use_condition, aa_mask, 0),
                      dssp_mask=jnp.where(use_condition, dssp_mask, 0))

        return result

    def apply_diffusion(self, data):
        c = self.config
        batch = data["batch_index"]
        mask = data["mask"]
        is_target = data["is_target"]
        hotspots = data["hotspots"]
        center_hotspot = data["center_hotspot"]
        pos = Vec3Array.from_array(data["pos"])
        t_pos = jax.random.uniform(
            hk.next_rng_key(), (batch.shape[0],))[batch]
        sigma = SIGMA_SCALE[c.diffusion_time_scale](t_pos)
        t_pos = sigma
        if "t_pos" in data:
            t_pos = data["t_pos"]
        cloud_std = c.sigma_data
        # diffuse positions centered at the target hotspots
        # TODO add option to use binder-centered instead
        if c.binder_centered:
            raise NotImplementedError("TODO: binder centered diffusion")
        else:
            pos_noised = diffuse_target_centered(
                hk.next_rng_key(), pos, is_target, center_hotspot,
                mask, batch, t_pos[:, None], cloud_std=cloud_std)
        return dict(
            pos_noised=pos_noised.to_array(),
            t_pos=t_pos)

    def __call__(self, data):
        c = self.config
        diffusion = Diffusion(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # prepare condition / task information
        data.update(self.prepare_condition(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))

        # self-conditioning
        def iteration_body(i, prev):
            override = self.apply_diffusion(data)
            result = diffusion(data, prev, override=override)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            return jax.lax.stop_gradient(prev)
        prev = diffusion.init_prev(data)
        if not hk.running_init():
           count = jax.random.randint(hk.next_rng_key(), (), 0, 2)
           prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = diffusion(data, prev)
        total, losses = diffusion.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

class BinderDiffusionInference(BinderDiffusion):
    def __init__(self, config,
                 name: Optional[str] = "structure_diffusion"):
        super().__init__(config, name)
        self.config = config

    def __call__(self, data, prev):
        c = self.config
        diffusion = Diffusion(c)

        # prepare condition / task information
        data.update(self.prepare_condition(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))
        # potentially process noised data
        if c.sym_noise is not None:
            data["pos_noised"] = c.sym_noise(data["pos_noised"])
        def apply_model(prev):
            result = diffusion(data, prev, predict_aa=True)
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

    def prepare_condition(self, data):
        c = self.config
        result = dict()
        cond = Condition(c)
        aa = 20 * jnp.ones_like(data["aa_gt"])
        if "aa_condition" in data:
            aa = data["aa_condition"]
        dssp = 3 * jnp.ones_like(data["aa_gt"])
        if "dssp_condition" in data:
            dssp = data["dssp_condition"]
        dssp_mean = jnp.zeros((aa.shape[0], 3), dtype=jnp.float32)
        dssp_mean_mask = jnp.zeros(aa.shape, dtype=jnp.bool_)
        if "dssp_mean" in data:
            dssp_mean = jnp.stack([data["dssp_mean"]] * aa.shape[0], axis=0)
            dssp_mean_mask = jnp.ones(aa.shape, dtype=jnp.bool_)
        condition, _, _ = cond(aa, dssp, data["mask"], data["residue_index"], data["chain_index"], data["batch_index"],
                            set_condition=dict(
                                aa=aa, dssp=dssp, dssp_mean=dssp_mean, dssp_mean_mask=dssp_mean_mask
                            ))
        result["condition"] = condition#jnp.zeros_like(condition)

        dmap = jnp.zeros((aa.shape[0], aa.shape[0]), dtype=jnp.float32)
        omap = jnp.zeros((aa.shape[0], aa.shape[0], 9), dtype=jnp.float32)
        dmap_mask = jnp.zeros_like(dmap, dtype=jnp.bool_)
        if "dmap_mask" in data:
            dmap = data["dmap"]
            omap = data["omap"]
            dmap_mask = data["dmap_mask"]

        chain = data["chain_index"]
        other_chain = chain[:, None] != chain[None, :]
        flags = jnp.concatenate((
            other_chain[..., None],
            jnp.zeros((chain.shape[0], chain.shape[0], 2)),
        ), axis=-1)
        pair_condition = dict(
            dmap=dmap,
            omap=omap,
            dmap_mask=dmap_mask,
            flags=flags#jnp.zeros_like(flags)
        )
        result["pair_condition"] = pair_condition
        return result

    def apply_diffusion(self, data):
        c = self.config
        batch = data["batch_index"]
        mask = data["mask"]
        pos = data["pos"]
        is_target = data["is_target"]
        hotspots = data["hotspots"]

        pos = Vec3Array.from_array(pos)
        t_pos = data["t_pos"]

        cloud_std = c.sigma_data
        pos_noised = diffuse_target_centered(
            hk.next_rng_key(), pos, is_target, hotspots,
            mask, batch, t_pos[:, None], cloud_std=cloud_std)
        return dict(pos_noised=pos_noised, t_pos=t_pos)

class BinderDiffusionNoise:
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

class BinderDiffusionEncode(BinderDiffusion):
    """Barebones encoder module to use as a component in a composite diffusion pipeline."""
    def __call__(self, data):
        target, _, pos = self.encode(data)
        return target, pos

class BinderDiffusionPredict(BinderDiffusionInference):
    """Barebones denoising module to use as a component in a composite diffusion pipeline."""
    def __call__(self, data, prev):
        c = self.config
        diffusion = Diffusion(c)

        # prepare condition / task information
        data.update(self.prepare_condition(data))
        def apply_model(prev):
            result = diffusion(data, prev, predict_aa=True)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            return result, prev

        if prev is None:
            prev = diffusion.init_prev(data)
        result, prev = apply_model(prev)
        return result, prev

class TargetEncoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "encoder"):
        super().__init__(name)
        self.config = config

    def prepare_features(self, data):
        c = self.config
        positions = data["pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        is_target = data["is_target"]
        hotspots = data["hotspots"]
        mask = data["mask"] * is_target
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
            jnp.log(dist + 1),
            is_target[:, None],
            hotspots[:, None]
        ]
        local = MLP(c.local_size * 4, c.local_size, activation=jax.nn.gelu,
                    bias=False)(
            jnp.concatenate(local_features, axis=-1))

        positions = positions.to_array()
        local = hk.LayerNorm([-1], True, True)(local)
        return local, positions, neighbours, resi, chain, batch, mask

    def __call__(self, data):
        c = self.config
        is_target = data["is_target"]
        local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data)
        local = EncoderStack(c, 3)(
            local, pos, neighbours, resi, chain, batch, mask)
        local = jnp.where(is_target[:, None], local, 0)
        return local

def get_target_asymmetric_neighbours(count: int):
    def inner(pos: Vec3Array, is_target, batch, mask, neighbours=None):
        distance = (pos[:, None] - pos[None, :]).norm()
        same_batch = batch[:, None] == batch[None, :]
        distance = jnp.where(same_batch, distance, jnp.inf)
        # invalidate all distances from the target to the binder
        # keeping all distances from the binder to the target.
        # This way, the binder (which gets noised), gets to learn
        # about the target, but not vice versa - the target only
        # sees neighbours within itself
        target_binder = is_target[:, None] * (~is_target)[None, :]
        distance = jnp.where(target_binder, jnp.inf, distance)
        mask = (mask[:, None] * mask[None, :] * same_batch)
        return get_neighbours(count)(distance, mask, neighbours)
    return inner

class Encoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "encoder"):
        super().__init__(name)
        self.config = config

    def prepare_features(self, data):
        c = self.config
        positions = data["pos"]
        # add noise to the encoder's input positions
        # to combat overfitting
        if not c.eval:
            positions += 0.2 * jax.random.normal(hk.next_rng_key(), positions.shape)
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        is_target = data["is_target"]
        hotspots = data["hotspots"]
        # add AA conditioning for the target
        aa = data["aa_gt"]
        aa = jnp.where(is_target, aa, 20)
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
        # get asymmetric neighbours:
        # the binder is allowed to know about the target
        # but NOT vice versa
        neighbours = get_target_asymmetric_neighbours(c.num_neighbours)(
            positions[:, 4], is_target, batch, mask)
        _, local_positions = extract_aa_frames(positions)
        dist = local_positions.norm()

        local_features = [
            local_positions.normalized().to_array().reshape(
                local_positions.shape[0], -1),
            distance_rbf(dist, 0.0, 22.0, 16).reshape(
                local_positions.shape[0], -1),
            jnp.log(dist + 1),
            is_target[:, None],
            hotspots[:, None],
            jax.nn.one_hot(aa, 21, axis=-1)
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
            local, pos, neighbours, resi, chain, chain, mask)
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

def preconditioning_scale_factors(config, t, sigma_data):
    return  {"in": 1.0, "out": sigma_data, "skip": 1.0,
             "refine": 1, "loss": 1}

class Diffusion(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config
        self.diffusion_block = BinderDiffusionBlock

    def __call__(self, data, prev, override=None, predict_aa=False):
        c = self.config
        self.diffusion_block = BinderDiffusionBlock
        diffusion_stack_module = DiffusionStack
        diffusion_stack = diffusion_stack_module(c, self.diffusion_block)
        aa_decoder = AADecoder(c)
        dssp_decoder = Linear(3, bias=False, initializer="zeros")
        distogram_decoder = DistogramDecoder(c)
        features = self.prepare_features(data, prev, override=override)
        local, pos, prev_pos, condition, target_condition, t_pos, resi, chain, batch, mask = features
        
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

        local, pos, trajectory = diffusion_stack(
            local, pos, prev_pos,
            condition, target_condition,
            t_pos, resi, chain, batch, mask,
            cyclic_mask=cyclic_mask,
            sup_neighbours=sup_neighbours)
        # predict & handle sequence, losses etc.
        result = dict()
        if c.distogram_trajectory:
            trajectory, distogram_trajectory = trajectory
            result["distogram_trajectory"] = distogram_trajectory
        elif c.aa_trajectory:
            trajectory, aa_trajectory = trajectory
            result["aa_trajectory"] = aa_trajectory
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
        result["pos_noised"] = data["pos_noised"]
        # dssp decoder features
        result["dssp"] = jax.nn.log_softmax(dssp_decoder(local))
        # distogram decoder features
        result["distogram_neighbours"] = distogram_neighbours
        result["distogram"] = distogram_decoder(local, pos, distogram_neighbours,
                                                resi, chain, batch, mask,
                                                cyclic_mask=cyclic_mask)
        # generate all-atom positions using predicted
        # side-chain torsion angles:
        aatype = data["aa_gt"]
        if predict_aa:
            if c.sample_aa:
                aatype = aa_decoder.sample(
                    data["aa_gt"], local, pos, resi, chain, batch, mask)
            else:
                aatype = jnp.argmax(aa_logits, axis=-1)
        if "aa_condition" in data:
            aatype = jnp.where(data["aa_condition"] != 20, data["aa_condition"], aatype)
        result["aatype"] = aatype
        raw_angles, angles, angle_pos = get_angle_positions(
            aatype, local, pos)
        result["raw_angles"] = raw_angles
        result["angles"] = angles
        result["atom_pos"] = angle_pos
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
        if override is not None:
            pos = override["pos_noised"]
            t_pos = override["t_pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        condition = data["condition"]
        target = data["target"]
        target_condition = dict(
            is_target=data["is_target"],
            hotspots=data["hotspots"],
            target=data["target"],
        )
        time_embedding = lambda x: jnp.concatenate((
            x[:, None],
            fourier_time_embedding(x, size=256)
        ), axis=-1)
        pos_time_features = time_embedding(t_pos)
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1),
            target
        ]
        if c.time_embedding:
            local_features = [pos_time_features] + local_features
        local_features.append(hk.LayerNorm([-1], True, True)(prev["local"]))
        local_features = jnp.concatenate(
            local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
        condition = hk.LayerNorm([-1], True, True)(condition)
        condition = MLP(
            2 * c.local_size, c.local_size,
            activation=jax.nn.gelu, bias=False,
            final_init=init_linear())(
                jnp.concatenate((condition, target), axis=-1))
        local = hk.LayerNorm([-1], True, True)(local)

        return local, pos, prev["pos"], condition, target_condition, t_pos, resi, chain, batch, mask

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
        aa_predict_mask = mask * (data["aa_gt"] != 20) * result["corrupt_aa"]
        # exclude target amino acids from AA prediction
        aa_predict_mask *= ~data["is_target"]
        aa_predict_mask = jnp.where(data["aa_mask"], 0, aa_predict_mask)
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
        dssp_predict_mask = mask * (1 - data["dssp_mask"])
        dssp_nll = -(result["dssp"] * jax.nn.one_hot(data["dssp_gt"], 3, axis=-1)).sum(axis=-1)
        dssp_nll = jnp.where(dssp_predict_mask, dssp_nll, 0)
        dssp_nll = dssp_nll.sum() / jnp.maximum(dssp_predict_mask.sum(), 1)
        losses["dssp"] = dssp_nll
        total += 1.0 * dssp_nll

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
        pos_mask = mask[..., None] * jnp.ones_like(pos_gt[..., 0], dtype=jnp.bool_)
        if c.mask_atom14:
            pos_mask *= data["atom_mask"]
        dist2 = ((result["pos"] - pos_gt) ** 2).sum(axis=-1)
        dist2 *= pos_mask
        dist2 = jnp.clip(dist2, 0, do_clip[:, None]).sum(axis=-1) / jnp.maximum(pos_mask.sum(axis=-1), 1)
        dist2 = jnp.where(mask, dist2, 0)
        pos_loss = (dist2 * diffusion_weight).sum() / 3
        losses["pos"] = pos_loss
        total += c.pos_weight * pos_loss
        # diffusion (z-space trajectory)
        dist2 = ((result["trajectory"] - pos_gt[None]) ** 2).sum(axis=-1)
        dist2 *= pos_mask
        dist2 = jnp.clip(dist2, 0, do_clip[None, :, None]).sum(axis=-1) / jnp.maximum(pos_mask.sum(axis=-1), 1)
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
            dd_mask = pair_mask[..., None, None] * pos_mask[:, None, :, None] * pos_mask[neighbours, None, :] * (dist_gt < 15.0)
            # same atom within residue
            same_atom = jnp.eye(pos_gt.shape[-1])[None, None]
            # AND same residue
            same_atom *= (neighbours == axis_index(neighbours, axis=0)[:, None])[..., None, None]
            dd_mask = dd_mask * (1 - same_atom) > 0
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

    def sample(self, aa, local, pos, resi, chain, batch, mask):
        c = self.config
        aa = 20 * jnp.ones_like(aa)
        def iter(i, carry):
            aa = carry
            logits, _ = self(
                aa, local, pos, resi, chain, batch, mask)
            logits = logits / (c.temperature or 0.1)
            index = jax.random.uniform(hk.next_rng_key(), aa.shape)
            index = jnp.where(aa == 20, index, jnp.inf)
            index = jnp.argmin(index, axis=0)
            aa_sample = jax.random.categorical(hk.next_rng_key(), logits, axis=-1)
            aa_new = aa.at[index].set(aa_sample[index])
            aa_new = jnp.where(aa == 20, aa_new, aa)
            return aa_new
        result = hk.fori_loop(0, aa.shape[0], iter, aa)
        return result
    
    def argmax(self, aa, local, pos, resi, chain, batch, mask):
        aa = 20 * jnp.ones_like(aa)
        logits, _ = self(
            aa, local, pos, resi, chain, batch, mask)
        return jnp.argmax(logits, axis=-1)

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
                 condition, target_condition, time,
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
                    condition, target_condition, time,
                    resi, chain, batch, mask,
                    cyclic_mask=cyclic_mask,
                    sup_neighbours=sup_neighbours)
                features, pos = result
                trajectory_output = pos
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
            trajectory = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        return local, pos, trajectory

def extract_condition_neighbours(num_index, num_spatial, num_random, num_block):
    def inner(pos, pair_condition, resi, chain, batch, mask):
        # neighbours by residue index
        neighbours = get_index_neighbours(num_index)(resi, chain, batch, mask)
        # get nearest neighbours by distance, using dmap information
        # where available
        ca = pos[:, 1]
        dist = (ca[:, None] - ca[None, :]).norm()
        dist = jnp.where(pair_condition["dmap_mask"], pair_condition["dmap"], dist)
        same_batch = batch[:, None] == batch[None, :]
        neighbours = get_neighbours(num_spatial)(
            dist, mask[:, None] * mask[None, :] * same_batch, neighbours)
        # get random neighbours based on distance
        neighbours = get_random_neighbours(num_random)(
            dist, batch, mask, neighbours)
        # get additional random neighbours defined by block conditioning
        block_contact_dist = jnp.where(
            pair_condition["flags"].any(axis=-1),
            pair_condition["flags"].any(axis=-1) * 1.0,
            jnp.inf)
        neighbours = get_random_neighbours(num_block)(block_contact_dist, batch, mask, neighbours)
        return neighbours
    return inner

def extract_target_neighbours(count):
    def inner(pos, is_target, hotspot, resi, chain, batch, mask):
        # get CB distances
        dist = jnp.linalg.norm(pos[:, None, 4] - pos[None, :, 4], axis=-1)
        # set all distances to non-target residues to inf
        dist = jnp.where(is_target[None, :], dist, jnp.inf)
        # set all distances to hotspot residues to 0
        dist = jnp.where(hotspot[None, :], 0, dist)
        same_batch = batch[:, None] == batch[None, :]
        pair_mask = same_batch * mask[:, None] * mask[None, :]
        # get nearest neighbours by target-hotspot distances
        neighbours = get_neighbours(count)(dist, pair_mask, None)
        return neighbours
    return inner

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

def binder_diffusion_pair_features(c):
    def inner(pos, target_condition, neighbours,
              resi, chain, batch, mask, cyclic_mask=None, init_pos=None):
        is_target = target_condition["is_target"]
        is_target = jax.nn.one_hot(is_target.astype(jnp.int32), 2)
        hotspots = target_condition["hotspots"]
        hotspots = jax.nn.one_hot(hotspots.astype(jnp.int32), 2)
        target_pair = is_target[:, None, :, None] * is_target[neighbours, None, :]
        target_pair = target_pair.reshape(*neighbours.shape, -1)
        hotspot_pair = is_target[:, None, :, None] * hotspots[neighbours, None, :]
        hotspot_pair = hotspot_pair.reshape(*neighbours.shape, -1)
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, cyclic=cyclic_mask is not None)(
                resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
        pos = Vec3Array.from_array(pos)
        # onehot features for binder-target and binder-hotspot contacts
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            jnp.concatenate((target_pair, hotspot_pair), axis=-1) )
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        if init_pos is not None:
            pos = Vec3Array.from_array(init_pos)
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                position_rotation_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

class BinderDiffusionBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "binder_diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, prev_pos,
                 condition, target_condition, time,
                 resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None, init_pos=None):
        c = self.config
        residual_update, residual_input, _ = get_residual_gadgets(
            features, c.resi_dual)
        current_neighbours = extract_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=32)(
                Vec3Array.from_array(pos),
                resi, chain, batch, mask)
        target_neighbours = extract_target_neighbours(32)(
            pos, target_condition["is_target"], target_condition["hotspots"],
            resi, chain, batch, mask)

        pair_feature_function = binder_diffusion_pair_features
        attention = SparseStructureAttention

        # condition the model on target features
        pair, pair_mask = pair_feature_function(c)(
            pos, target_condition, target_neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask, init_pos=prev_pos)
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), pos / c.sigma_data, pair, pair_mask,
                target_neighbours, resi, chain, batch, mask))
        # apply self-attention across all residues
        pair, pair_mask = pair_feature_function(c)(
            pos, target_condition, current_neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask, init_pos=prev_pos)
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), pos / c.sigma_data, pair, pair_mask,
                current_neighbours, resi, chain, batch, mask))
        # global update of local features
        features = residual_update(
            features,
            Update(c, global_update=True)(
                residual_input(features),
                pos, chain, batch, mask, condition))
        # normalise hidden representaiton for position update
        if c.resi_dual:
            local, incremental = features
            local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local_norm = residual_input(features)
        # update positions
        out_scale = preconditioning_scale_factors(
            c, time[:, None], c.sigma_data)["out"]
        position_update_function = update_positions
        if c.equivariance == "semi_equivariant":
            position_update_function = semiequivariant_update_positions
        pos = position_update_function(
            pos, local_norm,
            scale=out_scale,
            symm=c.symm)
        return features, pos.astype(local_norm.dtype)

def get_residual_gadgets(features, use_resi_dual=True):
    residual_update = (lambda x, y: resi_dual(*x, y)) if use_resi_dual else prenorm_skip
    residual_input = resi_dual_input if use_resi_dual else prenorm_input
    local_shape = features[0].shape[-1] if use_resi_dual else features.shape[-1]
    return residual_update, residual_input, local_shape

def make_pair_mask(mask, neighbours):
    return mask[:, None] * mask[neighbours] * (neighbours != -1)

class Condition(hk.Module):
    def __init__(self, config, name: Optional[str] = "condition"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, dssp, mask, residue_index, chain_index, batch_index, set_condition=None):
        sequence_mask, dssp_mask, dssp_mean_mask = self.get_masks(
            chain_index, batch_index, set_condition=set_condition)

        c = self.config
        aa_latent = Linear(c.local_size, initializer=init_glorot())(jax.nn.one_hot(aa, 21, axis=-1))
        aa_latent = hk.LayerNorm([-1], True, True)(aa_latent)

        dssp_latent = Linear(c.local_size, initializer=init_glorot())(jax.nn.one_hot(dssp, 3, axis=-1))
        dssp_latent = hk.LayerNorm([-1], True, True)(dssp_latent)

        dssp_mean = index_mean(jax.nn.one_hot(dssp, 3, axis=-1), chain_index, mask[:, None])
        dssp_mean /= jnp.maximum(dssp_mean.sum(axis=-1, keepdims=True), 1e-6)
        if (set_condition is not None) and (set_condition["dssp_mean"] is not None):
            dssp_mean = set_condition["dssp_mean"]
        dssp_mean = Linear(c.local_size, initializer=init_glorot())(dssp_mean)
        dssp_mean = hk.LayerNorm([-1], True, True)(dssp_mean)

        condition = jnp.where(dssp_mask[..., None], dssp_latent, 0) \
                  + jnp.where(dssp_mean_mask[..., None], dssp_mean, 0) \
                  + jnp.where(sequence_mask[..., None], aa_latent, 0)
        return condition, sequence_mask, dssp_mask

    def get_masks(self, chain_index, batch_index, set_condition=None):
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
