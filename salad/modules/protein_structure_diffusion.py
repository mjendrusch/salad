# This module explores trade-offs between different model
# architectures for the conditional and unconditional generation of
# protein backbones:
# - single-step structure denoising VS iterative refinement
# - invariant point attention VS MLP-based message passing
# - backbone-only (NCaCO) VS
#   augmented (NCaCO + learned pseudoatoms) structural features
# - Denoising Diffusion Probabilistic Models VS
#   Denoising Generative Models
# - different noise schedules: EDM-VE, VP, Chroma (Ingraham et al. 2021)
# - joint sequence design VS backbone only
# - auxiliary losses (FAPE, etc.) VS no auxiliary losses (only |x_\theta(x', t) - x|^2)

from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk

# alphafold dependencies
from salad.aflib.model.geometry import Vec3Array, Rigid3Array
from salad.aflib.model.all_atom_multimer import (
    get_atom37_mask, get_atom14_mask, atom37_to_atom14, atom14_to_atom37, get_alt_atom14, get_atom14_is_ambiguous, find_optimal_renaming,
    torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos)

# basic module imports
from salad.modules.basic import (
    Linear, MLP, GatedMLP, init_glorot, init_relu,
    init_zeros, init_linear, block_stack
)
from salad.modules.transformer import (
    resi_dual, prenorm_skip, resi_dual_input, prenorm_input)

# import geometry utils
from salad.modules.utils.geometry import (
    index_mean, index_sum, index_count, extract_aa_frames,
    extract_neighbours, distance_rbf, assign_sse,
    unique_chain, positions_to_ncacocb,
    single_protein_sidechains, compute_pseudo_cb,
    get_contact_neighbours)

from salad.modules.utils.dssp import assign_dssp, drop_dssp

# import violation loss
from salad.modules.utils.alphafold_loss import (
    violation_loss, all_atom_fape)

# sparse geometric module imports
from salad.modules.geometric import (
    SparseStructureMessage, SparseStructureAttention,
    SparseInvariantPointAttention,
    VectorLinear, VectorMLP, VectorLayerNorm, LinearToPoints,
    vector_mean_norm, vector_std_norm,
    frame_pair_features, frame_pair_vector_features,
    sequence_relative_position, equivariant_pair_embedding,
    sum_equivariant_pair_embedding,
    distance_features, rotation_features
)

# diffusion processes
from salad.modules.utils.diffusion import (
    diffuse_coordinates_edm, diffuse_coordinates_vp, diffuse_coordinates_blend,
    diffuse_coordinates_vp_scaled, diffuse_coordinates_chroma,
    diffuse_features_edm, diffuse_features_vp, diffuse_atom_cloud,
    diffuse_atom_chain_cloud,
    diffuse_sequence, log_sigma_embedding, fourier_time_embedding
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
        residue_gt = data["aa_gt"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]

        # if positions come in the atom24 format,
        # which is meant for all AA + all NA residues,
        # recast them into atom37:
        if pos.shape[1] == 24:
            # first, truncate to atom14 format
            pos = pos[:, :14]
            atom_mask = atom_mask[:, :14]
            # then, convert to atom37 format
            pos = atom14_to_atom37(pos, residue_gt)
            atom_mask = atom14_to_atom37(atom_mask, residue_gt)

        # uniquify chain IDs across batches:
        chain = unique_chain(chain, batch)
        mask = data["seq_mask"] * data["residue_mask"]        
        
        # subtract the center from all positions
        center = index_mean(pos[:, 1], batch, atom_mask[:, 1, None])
        pos = pos - center[:, None]
        # set the positions of all masked atoms to the pseudo Cb position
        pos = jnp.where(
            atom_mask[..., None], pos,
            compute_pseudo_cb(pos)[:, None, :])

        # convert positions to atom14 format for convenience
        pos_37 = pos
        atom_mask_37 = atom_mask
        pos_37 = jnp.where(
            atom_mask_37[..., None], pos_37,
            compute_pseudo_cb(pos_37)[:, None, :])
        pos_14, atom_mask_14 = atom37_to_atom14(
            residue_gt, Vec3Array.from_array(pos_37), atom_mask_37)
        pos_14 = pos_14.to_array()
        # FIXME: this is new and necessary, otherwise masked
        # positions get reset to 0 by atom37_to_atom14
        # and all alanines become cursed (pseudo-Cb becomes
        # 0, 0, 0 and is used everywhere as a garbage value)
        pos_14 = jnp.where(
            atom_mask_14[..., None], pos_14,
            compute_pseudo_cb(pos_14)[:, None, :])
        pos = pos_14
        atom_mask = atom_mask_14

        # if specified, augment backbone with encoded positions
        data["chain_index"] = chain
        data["pos"] = pos
        data["atom_mask"] = atom_mask
        data["mask"] = mask
        seq, pos = self.encode(data)

        # set all-atom-position target
        if c.output_structure_kind in ("angle", "atom14"):
            atom_pos = pos_14
            atom_mask = atom_mask_14
        elif c.output_structure_kind == "atom37":
            atom_pos = pos_37
            atom_mask = atom_mask_37
        else:
            atom_pos = positions_to_ncacocb(pos)
            atom_mask = jnp.concatenate(
                (atom_mask_14[:, :4], jnp.ones_like(atom_mask_14[:, 0:1])), axis=-1)
        return dict(seq=seq, pos=pos, chain_index=chain, mask=mask,
                    atom_pos=atom_pos, atom_mask=atom_mask,
                    all_atom_positions=pos_37,
                    all_atom_mask=atom_mask_37)

    def encode(self, data):
        pos = data["pos"]
        aatype = data["aa_gt"]
        c = self.config
        if c.input_structure_kind == "backbone":
            # set up ncaco + pseudo cb positions
            backbone = positions_to_ncacocb(pos)
            pos = backbone
        elif c.input_structure_kind == "atom14":
            # keep atom14 positions
            pos = pos
        elif c.input_structure_kind == "atom37":
            # switch to atom37 positions
            pos = atom14_to_atom37(pos, aatype)
            atom_mask = atom14_to_atom37(atom_mask, aatype)
            pos = jnp.where(atom_mask[..., None], pos, pos[:, 1:2])
        data["pos"] = pos
        # add additional vector or scalar features
        augmented_seq, augmented_pos = Encoder(c)(data)
        if c.augment_size:
            pos = augmented_pos
        
        seq = jax.nn.one_hot(aatype, 20, axis=-1)
        seq = seq
        seq /= jnp.maximum(seq.sum(axis=-1, keepdims=True), 1e-3)
        if c.sequence_kind == "linear":
            seq = Linear(c.seq_size, bias=False)(seq)
            seq = hk.LayerNorm([-1], False, False)(seq)
        elif c.sequence_kind == "onehot":
            seq = c.seq_factor * seq
        else:
            seq = Linear(c.seq_size, bias=False)(augmented_seq)
            seq = hk.LayerNorm([-1], False, False)(seq)
        return seq, pos

    def prepare_condition(self, data):
        c = self.config
        aa = data["aa_gt"]
        pos = data["all_atom_positions"]
        pos_mask = data["all_atom_mask"]
        dssp, blocks, block_adjacency = assign_dssp(pos, data["batch_index"], pos_mask.any(axis=-1))
        _, block_adjacency = drop_dssp(
            hk.next_rng_key(), dssp, blocks, block_adjacency, p_drop=c.p_drop_dssp)
        condition, aa_mask = Condition(c)(
            aa, dssp, pos, pos_mask,
            data["residue_index"], data["chain_index"],
            data["batch_index"], do_condition=None)
        # include conditioning stochastically during training
        coin = jax.random.bernoulli(hk.next_rng_key(), 0.5, (aa.shape[0],))
        coin = coin[data["batch_index"]]
        # if c.eval: # FIXME
        #     coin = jnp.zeros_like(coin)
        condition = jnp.where(coin[..., None], condition, 0)
        result = dict(condition=condition, aa_mask=jnp.where(coin, aa_mask, 0))
        if c.pair_condition:
            pair_condition = get_pair_condition(pos, aa, block_adjacency, pos_mask,
                                                data["residue_index"], data["chain_index"], data["batch_index"],
                                                do_condition=None)
            pair_condition = jnp.where((coin[:, None] * coin[None, :])[..., None], pair_condition, 0)
            result["pair_condition"] = pair_condition
        return result

    def apply_diffusion(self, data):
        c = self.config
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        pos = Vec3Array.from_array(data["pos"])
        seq = data["seq"]
        if c.diffusion_kind == "edm":
            sigma_pos, randomness_pos = get_sigma_edm(
                batch,
                meanval=c.pos_mean_sigma,
                stdval=c.pos_std_sigma,
                minval=c.pos_min_sigma,
                maxval=c.pos_max_sigma)
            # FIXME: turn off noise
            # sigma_pos = 0.05 * jnp.ones_like(sigma_pos)
            pos_noised = diffuse_coordinates_edm(
                hk.next_rng_key(), pos, batch, sigma_pos)
            randomness_seq = None
            if c.tie_noiselevels:
                randomness_seq = randomness_pos
            sigma_seq, randomness_seq = get_sigma_edm(
                batch,
                meanval=c.seq_mean_sigma,
                stdval=c.seq_std_sigma,
                minval=c.seq_min_sigma,
                maxval=c.seq_max_sigma,
                randomness=randomness_seq)
            seq_noised = diffuse_features_edm(
                hk.next_rng_key(), seq, sigma_seq)
            t_pos = sigma_pos
            t_seq = sigma_seq
        if c.diffusion_kind == "flow":
            t_pos = jax.random.uniform(
                hk.next_rng_key(), (batch.shape[0], 1))[batch]
            if c.tie_noiselevels:
                t_seq = t_pos
            else:
                t_seq = jax.random.uniform(
                    hk.next_rng_key(), (batch.shape[0], 1))
            pos_noised = diffuse_coordinates_blend(
                hk.next_rng_key(), pos, mask, batch, t_pos, scale=c.sigma_data)
            seq_noised = seq # TODO
        if c.diffusion_kind in ("fedm"):
            sigma_pos, randomness_pos = get_sigma_framediff_edm(
                batch)
            sigma_pos *= c.sigma_data
            pos_noised = diffuse_coordinates_edm(
                hk.next_rng_key(), pos, batch, sigma_pos)
            randomness_seq = None
            if c.tie_noiselevels:
                randomness_seq = randomness_pos
            sigma_seq, randomness_seq = get_sigma_framediff_edm(
                batch, randomness=randomness_seq)
            seq_noised = diffuse_features_edm(
                hk.next_rng_key(), seq, sigma_seq)
            t_pos = sigma_pos
            t_seq = sigma_seq
        if c.diffusion_kind in ("vp", "vpfixed"):
            t_pos = jax.random.uniform(
                hk.next_rng_key(), (batch.shape[0], 1))[batch]
            sigma = SIGMA_SCALE[c.diffusion_time_scale](t_pos)
            t_pos = sigma
            if c.tie_noiselevels:
                t_seq = t_pos
            else:
                t_seq = jax.random.uniform(
                    hk.next_rng_key(), (batch.shape[0], 1))
            cloud_std = None
            if "cloud_std" in data:
                cloud_std = data["cloud_std"]
            if c.diffusion_kind == "vpfixed":
                cloud_std = c.sigma_data
            if c.correlated_cloud:
                pos_noised = diffuse_atom_chain_cloud(
                    hk.next_rng_key(), pos, mask, chain, batch, t_pos)
            else:
                pos_noised = diffuse_atom_cloud(
                    hk.next_rng_key(), pos, mask, batch, t_pos, cloud_std=cloud_std)
            seq_noised = diffuse_features_vp(
                hk.next_rng_key(), seq, t_seq,
                scale=c.sequence_noise_scale)
        return dict(
            seq_noised=seq_noised, pos_noised=pos_noised.to_array(),
            t_pos=t_pos, t_seq=t_seq)

    def __call__(self, data):
        c = self.config
        diffusion = Diffusion(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # prepare condition / task information
        data.update(self.prepare_condition(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))

        result = diffusion(data)
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

    def __call__(self, data):
        c = self.config
        diffusion = Diffusion(c)

        # prepare condition / task information
        data.update(self.prepare_condition(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))

        result = diffusion(data, predict_aa=True)

        # NOTE: also return noisy input for diagnostics
        result["pos_input"] = data["pos_noised"]
        
        return result

    def prepare_condition(self, data):
        c = self.config
        # TODO: allow non-zero condition
        result = dict(condition=jnp.zeros((data["pos"].shape[0], c.local_size)))
        cond = Condition(c)
        aa = jnp.zeros_like(data["aa_gt"])
        aa_mask = jnp.zeros(aa.shape[0], dtype=jnp.bool_)
        if "aa_condition" in data:
            aa = data["aa_condition"]
            aa_mask = aa != 20
        dssp = jnp.zeros_like(data["aa_gt"])
        dssp_mask = jnp.zeros(aa.shape[0], dtype=jnp.bool_)
        if "dssp_condition" in data:
            dssp = data["dssp_condition"]
            dssp_mask = dssp != 3
        dssp_mean = jnp.zeros((aa.shape[0], 3), dtype=jnp.float32)
        dssp_mean_mask = jnp.zeros(aa.shape, dtype=jnp.bool_)
        if "dssp_mean" in data:
            dssp_mean = jnp.repeat(data["dssp_mean"], index_count(data["chain_index"], data["mask"]), axis=0)
        # TODO
        # condition = cond(aa, dssp, jnp.zeros((aa.shape[0], 14, 3), dtype=jnp.float32), jnp.zeros((aa.shape[0], 14), dtype=jnp.bool_),
        #                  data["residue_index"], data["chain_index"], data["batch_index"], do_condition=dict(
        #                      aa=aa, dssp=dssp, dssp_mean=dssp_mean, dssp_mean_mask=dssp_mean_mask
        #                  ))

        if c.pair_condition:
            chain = data["chain_index"]
            other_chain = chain[:, None] != chain[None, :]
            pair_condition = jnp.concatenate((
                other_chain[..., None],
                jnp.zeros((chain.shape[0], chain.shape[0], 3)),
            ), axis=-1)
            # FIXME: make this input dependent
            # relative_hotspot = jnp.zeros_like(other_chain)
            # relative_hotspot = relative_hotspot.at[:, 15:20].set(1)
            # relative_hotspot = jnp.where(other_chain, relative_hotspot, 0)
            # pair_condition = pair_condition.at[:, :, 1].set(relative_hotspot)
            # pair_condition = pair_condition.at[:, :, 2].set(relative_hotspot.T)
            result["pair_condition"] = pair_condition
        return result

    def apply_diffusion(self, data):
        c = self.config
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        pos = data["pos"] - index_mean(data["pos"][:, 1], batch, data["mask"][..., None])[:, None]
        pos = Vec3Array.from_array(pos)
        seq = data["seq"]
        # drop_key = hk.next_rng_key() # FIXME
        if c.diffusion_kind in ("edm", "fedm"):
            sigma_pos = data["t_pos"]
            pos_noised = diffuse_coordinates_edm(
                hk.next_rng_key(), pos, batch, sigma_pos, symm=c.symm)
            sigma_seq = data["t_seq"]
            seq_noised = diffuse_features_edm(
                hk.next_rng_key(), seq, sigma_seq)
            t_pos = sigma_pos
            t_seq = sigma_seq
        if c.diffusion_kind == "flow":
            t_pos = data["t_pos"]
            t_seq = data["t_seq"]
            pos_noised = diffuse_coordinates_blend(
                hk.next_rng_key(), pos, mask, batch, t_pos, scale=c.sigma_data)
            seq_noised = seq
        if c.diffusion_kind in ("vp", "vpfixed"):
            # diffuse_coordinates = diffuse_atom_cloud
            scale = None
            t_pos = data["t_pos"]
            t_seq = data["t_seq"]
            sigma = SIGMA_SCALE[c.diffusion_time_scale](t_pos)
            # s = 0.01
            # alpha_bar = jnp.cos((t_pos + s) / (1 + s) * jnp.pi / 2) ** 2
            # alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
            # alpha_bar = jnp.clip(alpha_bar, 0, 1)
            # sigma = jnp.sqrt(1 - alpha_bar)
            t_pos = sigma
            cloud_std = None
            if "cloud_std" in data:
                cloud_std = data["cloud_std"][:, None, None]
            complex_std = cloud_std
            if "complex_std" in data:
                complex_std = data["complex_std"][:, None, None]
            if c.diffusion_kind == "vpfixed":
                cloud_std = c.sigma_data
            if c.correlated_cloud:
                pos_noised = diffuse_atom_chain_cloud(
                    hk.next_rng_key(), pos, mask, chain, batch, t_pos,
                    cloud_std=cloud_std, complex_std=complex_std, symm=c.symm)
            else:
                pos_noised = diffuse_atom_cloud(
                    hk.next_rng_key(), pos, mask, batch, t_pos,
                    cloud_std=cloud_std, symm=c.symm)
            seq_noised = diffuse_features_vp(
                hk.next_rng_key(), seq, t_seq,
                scale=c.sequence_noise_scale)
        if c.sde:
            pos_noised = Vec3Array.from_array(data["pos_noised"])
        return dict(
            seq_noised=seq_noised, pos_noised=pos_noised.to_array(),
            t_pos=t_pos, t_seq=t_seq)

def get_sigma_edm(batch, meanval=None, stdval=None, minval=0.01, maxval=80.0, randomness=None):
    size = batch.shape[0]
    if meanval is not None:
        # if we have specified a log mean & standard deviation
        # for a sigma distribution, sample sigma from that distribution
        if randomness is None:
            randomness = jax.random.normal(hk.next_rng_key(), (size, 1))[batch]
        log_sigma = meanval + stdval * randomness
        sigma = jnp.exp(log_sigma)
    else:
        # sample from a uniform distribution - this is suboptimal
        # but easier to tune
        if randomness is None:
            randomness = jax.random.uniform(
                hk.next_rng_key(), (size, 1),
                minval=0, maxval=1)[batch]
        sigma = minval + randomness * (maxval - minval)
    return sigma, randomness

def get_sigma_framediff_edm(batch, bmin=0.1, bmax=20.0, randomness=None):
    size = batch.shape[0]
    if randomness is None:
        randomness = jax.random.uniform(hk.next_rng_key(), (size, 1), minval=0.01, maxval=0.999)[batch]
    Gs = randomness * bmin + 0.5 * randomness ** 2 * (bmax - bmin)
    sigma = jnp.sqrt(1 - jnp.exp(-Gs))
    alpha = jnp.sqrt((1 - sigma**2))
    return sigma / alpha, randomness

# what types of diffusion models do we have?
# EDM - pre-scales the input, post-scales the output
# refinement (carries a trajectory) vs single-prediction (outputs one update, which is then scaled)
# presence & absence of different auxiliary losses
#
# the difficulty comes from EDM + refinement models
# here, we're updating & rescaling an ever-changing structure
# while still keeping track of the original input structure.
# This necessitates a two-pronged approach to rescaling
# and updating the output positions:
# the current updated position is rescaled by 1 / sigma_data _at each step_
# the initial position is rescaled by 1 / sqrt(sigma_data^2 + sigma_noise^2)
# both initial and updated positions are used in attention updates

class Encoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "encoder"):
        super().__init__(name)
        self.config = config

    def prepare_features(self, data):
        c = self.config
        residues = data["aa_gt"]
        positions = data["pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        positions = Vec3Array.from_array(positions)
        if c.pre_augment:
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
        neighbours = extract_neighbours(
            num_index=16, num_spatial=16, num_random=32)(
                positions, resi, chain, batch, mask)
        _, local_positions = extract_aa_frames(positions)
        dist = local_positions.norm()
        local_features = [
            local_positions.normalized().to_array().reshape(
                local_positions.shape[0], -1),
            distance_rbf(dist, 0.0, 22.0, 16).reshape(
                local_positions.shape[0], -1),
            jnp.log(dist + 1)
        ]
        if c.pre_augment:
            local_features = [
                local_positions.to_array().reshape(
                    local_positions.shape[0], -1)
            ]
        if c.embed_residues:
            local_features.append(jax.nn.one_hot(residues, 21, axis=-1))
        local = MLP(c.local_size * 4, c.local_size, activation=jax.nn.gelu,
                    bias=False)(
            jnp.concatenate(local_features, axis=-1))
        
        # FIXME: sum pair embedding
        pair, pair_mask = sum_equivariant_pair_embedding(c, use_local=True)(
            local, positions, neighbours, resi, chain, batch, mask)
        positions = positions.to_array()
        if not c.pre_augment:
            local += SparseStructureMessage(c)(
                local, positions, pair, pair_mask,
                neighbours, resi, chain, batch, mask)
        local = hk.LayerNorm([-1], True, True)(local)
        return local, positions, neighbours, resi, chain, batch, mask

    def __call__(self, data):
        c = self.config
        augment_size = c.augment_size
        local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data)
        local = EncoderStack(c, c.encoder_depth)(
            local, pos, neighbours, resi, chain, batch, mask)
        # generate augmented vector features from encoder representation
        frames, local_positions = extract_aa_frames(
            Vec3Array.from_array(pos))
        if c.pre_augment:
            augment = local_positions[:, 5:]
            update = MLP(2 * local.shape[-1], augment.shape[1] * 3,
                activation=jax.nn.gelu, final_init=init_linear())(local)
            update = update.reshape(update.shape[0], augment.shape[1], 3)
            update = Vec3Array.from_array(update)
            augment += update
            local_positions = local_positions[:, :5]
        else:
            augment = Vec3Array.from_array(LinearToPoints(
                augment_size, init=init_glorot())(local, frames))
            augment = frames[..., None].apply_inverse_to_point(augment)
            augment += VectorLinear(
                augment_size, initializer=init_glorot())(local_positions)
            augment += VectorMLP(
                augment_size * 2,
                augment_size,
                activation=jax.nn.gelu,
                final_init=init_glorot())(local, augment)
        if c.vector_norm == "layer_norm":
            augment = VectorLayerNorm()(augment)
        elif c.vector_norm == "std_norm":
            augment = vector_std_norm(augment)
        else:
            augment = vector_mean_norm(augment)
        # combine augmented features with backbone / atom positions
        local_positions = jnp.concatenate((
            local_positions.to_array(), augment.to_array()), axis=-2)
        pos = frames[:, None].apply_to_point(
            Vec3Array.from_array(local_positions))
        return local, pos.to_array()

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
        # extract residue local atom positions
        _, local_positions = extract_aa_frames(Vec3Array.from_array(pos))
        # add position information to local features
        features = residual_update(
            features,
            PositionToLocal(c.local_size)(local_positions.to_array())
        )
        # global update of local features
        if c.global_update:
            features = residual_update(
                features,
                GlobalUpdate(c)(residual_input(features), chain, batch, mask))
        # sparse structure message or attention
        # choose if we want to use attention
        attention_module = (
            SparseStructureAttention
            if c.use_attention
            else SparseStructureMessage
        )
        # embed pair features
        # FIXME: sum pair embedding
        pair, pair_mask = sum_equivariant_pair_embedding(
            c, use_local=not c.use_attention)(
                residual_input(features), Vec3Array.from_array(pos),
                neighbours, resi, chain, batch, mask)
        # attention / message passing module
        features = residual_update(
            features,
            attention_module(c)(
                residual_input(features), pos, pair, pair_mask,
                neighbours, resi, chain, batch, mask))
        # local feature transition (always enabled)
        features = residual_update(
            features,
            MLP(local_shape * c.factor, local_shape,
                activation=jax.nn.gelu, bias=False,
                final_init=init_zeros())(
                residual_input(features)))
        return features

class EncoderStack(hk.Module):
    def __init__(self, config, depth=None, name: Optional[str] = "self_cond_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 2

    def __call__(self, local, pos, fixed_neighbours, resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                neighbours = extract_neighbours(num_random=32)(
                    Vec3Array.from_array(pos), resi, chain, batch, mask)
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
        residual_update, residual_input, local_shape = get_residual_gadgets(features, c.resi_dual)
        # extract residue local atom positions
        _, local_positions = extract_aa_frames(Vec3Array.from_array(pos))
        # add position information to local features
        features = residual_update(
            features,
            PositionToLocal(c.local_size)(local_positions.to_array())
        )
        # global update of local features
        if c.global_update:
            features = residual_update(
                features,
                GlobalUpdate(c)(residual_input(features), chain, batch, mask))
        # sparse structure message or attention
        # choose if we want to use attention
        attention_module = (
            SparseStructureAttention
            if c.use_attention
            else SparseStructureMessage
        )
        # embed pair features
        pair_input_features = residual_input(features)
        pair_input_features += hk.LayerNorm([-1], True, True)(
            Linear(pair_input_features.shape[-1], bias=False)(
                jax.nn.one_hot(aa, 21)))
        # FIXME: sum pair embedding
        pair, pair_mask = sum_equivariant_pair_embedding(
            c, use_local=not c.use_attention)(
                pair_input_features, Vec3Array.from_array(pos),
                neighbours, resi, chain, batch, mask)
        if priority is not None:
            pair_mask *= priority[:, None] > priority[neighbours]
        # attention / message passing module
        features = residual_update(
            features,
            attention_module(c)(
                residual_input(features), pos, pair, pair_mask,
                neighbours, resi, chain, batch, mask))
        # local feature transition (always enabled)
        features = residual_update(
            features,
            MLP(local_shape * c.factor, local_shape, activation=jax.nn.gelu, bias=False, final_init=init_zeros())(
                residual_input(features)))
        return features

class AADecoderStack(hk.Module):
    def __init__(self, config, depth=None, name: Optional[str] = "aa_decoder_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 2

    def __call__(self, aa, local, pos, neighbours, resi, chain, batch, mask, priority=None):
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

def vp_edm_scaling(sigma_data, sigma):
    alpha = jnp.sqrt(1 - sigma ** 2)
    beta = sigma
    return edm_scaling(sigma_data, sigma_data, alpha=alpha, beta=beta)

def preconditioning_scale_factors(config, t, sigma_data):
    c = config
    if c.preconditioning == "edm":
        result = edm_scaling(sigma_data, 1.0, alpha=1.0, beta=t)
    elif c.preconditioning == "vpedm":
        result = vp_edm_scaling(sigma_data, t)
    elif c.preconditioning == "flow":
        result = {"in": 1.0, "out": sigma_data, "skip": 1.0,
                  "refine": 1, "loss": 1 / jnp.maximum(t, 0.01) ** 2}
    elif c.preconditioning == "fscale":
        result = {"in": 1.0, "out": sigma_data, "skip": t / jnp.sqrt(t ** 2 + sigma_data ** 2),
                  "refine": 1, "loss": 1}
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
        self.diffusion_blocks = {
            "old": RecomputedDiffusionBlock,
            "edm": EDMDiffusionBlock,
            "vp": VPDiffusionBlock
        }

    def __call__(self, data, predict_aa=False):
        c = self.config
        diffusion_block = self.diffusion_blocks[c.block_type]
        diffusion_stack = DiffusionStack(c, diffusion_block)
        aa_decoder = AADecoder(c)
        features = self.prepare_features(data)
        local, seq, pos, condition, t_pos, t_seq, resi, chain, batch, mask, pair_condition = features
        
        # enable selective cyclisation of chains
        cyclic_mask = None
        if c.cyclic and "cyclic_mask" in data:
            cyclic_mask = data["cyclic_mask"]

        initial_pos = None
        if c.use_initial_position:
            initial_pos = pos
        pos_factors = preconditioning_scale_factors(c, t_pos, c.sigma_data)
        seq_factors = preconditioning_scale_factors(c, t_seq, c.seq_factor)
        skip_scale_pos = pos_factors["skip"]
        if c.preconditioning in ("edm", "vpedm"):
            # FIXME: is this an issue?
            if c.scale_to_chain_center:
                pos = scale_to_center(pos, t_pos, skip_scale_pos, chain, mask)
            else:
                pos *= skip_scale_pos[..., None]
        local, pos, trajectory = diffusion_stack(
            local, pos, initial_pos, condition,
            t_pos, resi, chain, batch, mask,
            pair_condition=pair_condition,
            cyclic_mask=cyclic_mask)
        # predict & handle sequence, losses etc.
        result = dict()
        # trajectory
        if isinstance(trajectory, tuple):
            trajectory, feature_trajectory = trajectory
            result["feature_trajectory"] = feature_trajectory
        result["trajectory"] = trajectory
        # position update
        if not c.refine:
            if c.consensus_iterations:
                pos = update_positions_consensus(
                    pos, local, resi, chain, batch,
                    mask, scale=c.sigma_data,
                    num_iterations=c.consensus_iterations)
            else:
                pos = update_positions(pos, local, pos_factors["out"])
        # if c.symm:
        #     mean_pos = ((pos[:100] + 14) + pos[100:200] + (pos[200:] - 14)) / 3
        #     pos = jnp.concatenate((mean_pos - 14, mean_pos, mean_pos + 14), axis=0)
        result["pos"] = pos
        # sequence feature update
        seq_update = MLP(
            local.shape[-1] * 2, seq.shape[-1],
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_zeros())(local)
        seq = seq_factors["skip"] * seq + seq_factors["out"] * seq_update
        result["seq"] = seq

        # decoder features and logits
        aa_logits, decoder_features, corrupt_aa = aa_decoder.train(
            data["aa_gt"], local, pos, resi, chain, batch, mask)
        result["aa"] = aa_logits
        result["aa_features"] = decoder_features
        result["corrupt_aa"] = corrupt_aa * data["mask"] * (data["aa_gt"] != 20)
        result["aa_gt"] = data["aa_gt"]
        result["pos_noised"] = data["pos_noised"]
        # generate all-atom positions
        if c.output_structure_kind == "angle":
            # generate all-atom positions using predicted
            # side-chain torsion angles:
            aatype = data["aa_gt"]
            if predict_aa:
                aatype = jnp.argmax(aa_logits, axis=-1)
            raw_angles, angles, angle_pos = get_angle_positions(
                aatype, local, pos)
            _, _, angle_trajectory = jax.vmap(
                get_angle_positions, in_axes=(None, None, 0))(
                aatype, local, trajectory)
            result["raw_angles"] = raw_angles
            result["angles"] = angles
            result["atom_pos"] = angle_pos
            result["atom_trajectory"] = angle_trajectory
        elif c.output_structure_kind == "atom14":
            # all-atom positions are the first 14 latent positions
            result["atom_pos"] = pos[:, :14]
        elif c.output_structure_kind == "atom37":
            # all-atom positions are the first 37 latent positions
            result["atom_pos"] = pos[:, :37]
        else:
            # just deal with NCaCOCb coordinates
            result["atom_pos"] = pos[:, :5]
        return result

    def prepare_features(self, data):
        c = self.config
        pos = data["pos_noised"]
        t_pos = data["t_pos"]
        seq = data["seq_noised"]
        t_seq = data["t_seq"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        condition = data["condition"]
        pair_condition = None
        if c.pair_condition:
            pair_condition = data["pair_condition"]
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
        pos_time_features = time_embedding(t_pos[:, 0])
        frames, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1)
        ]
        if c.time_embedding:
            local_features = [pos_time_features] + local_features
        if c.diffuse_sequence:
            seq_factors = preconditioning_scale_factors(
                c, t_seq, c.seq_factor)
            seq_time_features = time_embedding(t_seq[:, 0])
            local_features += [
                seq_time_features,
                seq * seq_factors["in_scale"]
            ]
        local_features = jnp.concatenate(
            local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
        local = hk.LayerNorm([-1], True, True)(local)
        condition = hk.LayerNorm([-1], True, True)(condition)

        return local, seq, pos, condition, t_pos, t_seq, resi, chain, batch, mask, pair_condition

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
            late_mask = data["t_pos"][:, 0] < 2.0
        else:
            late_mask = data["t_pos"][:, 0] < 0.5
        late_mask *= mask
        
        # AA NLL loss
        if c.aa_weight:
            aa_predict_mask = mask * (data["aa_gt"] != 20) * result["corrupt_aa"]
            aa_predict_mask = jnp.where(data["aa_mask"], 0, aa_predict_mask)
            # if c.aa_decoder_kind == "adm":
            #     aa_predict_weight = 1 / jnp.maximum(index_sum(aa_predict_mask, batch, mask), 1e-6)
            #     aa_predict_weight /= batch.max() + 1
            # else:
            aa_predict_weight = 1 / jnp.maximum(aa_predict_mask.sum(keepdims=True), 1)
            aa_predict_weight = jnp.where(aa_predict_mask, aa_predict_weight, 0)
            aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
            aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
            aa_nll = (aa_nll * aa_predict_weight).sum()
            losses["aa"] = aa_nll
            total += c.aa_weight * aa_nll

        # diffusion losses
        base_weight = mask / jnp.maximum(index_sum(mask.astype(jnp.float32), batch, mask), 1) / (batch.max() + 1)
        # seq diffusion loss
        if c.seq_weight:
            sigma = data["t_seq"][:, 0]
            loss_weight = preconditioning_scale_factors(
                c, sigma, c.seq_factor)["loss"]
            seq_loss = ((result["seq"] - data["seq"]) ** 2).mean(axis=-1)
            seq_loss = (seq_loss * base_weight * loss_weight).sum()
            losses["seq"] = seq_loss
            total += c.seq_weight * seq_loss
        # seq decoder feature loss
        if c.decoder_weight:
            decoder_loss = ((result["aa_features"] - data["seq"]) ** 2).mean(axis=-1)
            decoder_loss = (decoder_loss * base_weight).sum()
            losses["decoder"] = decoder_loss
            total += c.decoder_weight * decoder_loss
        # diffusion (z-space)
        do_clip = jnp.where(
            jax.random.bernoulli(hk.next_rng_key(), c.p_clip, batch.shape)[batch],
            100.0,
            jnp.inf)
        if c.pos_weight:
            sigma = data["t_pos"][:, 0]
            loss_weight = preconditioning_scale_factors(
                c, sigma, c.sigma_data)["loss"]
            diffusion_weight = base_weight * loss_weight
            dist2 = ((result["pos"] - data["pos"]) ** 2).sum(axis=-1)
            dist2 = jnp.clip(dist2, 0, do_clip[:, None]).mean(axis=-1) # FIXME: changed order of mean and sum
            pos_loss = (dist2 * diffusion_weight).sum() / 3
            losses["pos"] = pos_loss
            total += c.pos_weight * pos_loss
        # diffusion (z-space trajectory)
        if c.trajectory_weight and c.refine and not c.no_trajectory_loss:
            trajectory_index = (result["trajectory"].shape[0] - 1) - jnp.arange(result["trajectory"].shape[0])
            trajectory_weight = c.trajectory_discount ** trajectory_index
            trajectory_weight /= trajectory_weight.sum()
            dist2 = ((result["trajectory"] - data["pos"][None]) ** 2).sum(axis=-1)
            dist2 = jnp.clip(dist2, 0, do_clip[None, :, None]).mean(axis=-1)
            trajectory_pos_loss = (dist2 * diffusion_weight[None, ...]).sum(axis=1) / 3
            trajectory_pos_loss = (trajectory_pos_loss * trajectory_weight).sum()
            unweighted_trajectory_pos_loss = (dist2 * base_weight[None, ...]).sum(axis=1) / 3
            unweighted_trajectory_pos_loss = (unweighted_trajectory_pos_loss * trajectory_weight).sum()
            losses["pos_trajectory"] = trajectory_pos_loss
            losses["pos_trajectory_unweighted"] = unweighted_trajectory_pos_loss
            
            # FIXME: trajectory x-loss without weighting
            total += c.trajectory_weight * (trajectory_pos_loss + unweighted_trajectory_pos_loss)
        # diffusion (x-space)
        if c.x_weight:
            dist2 = ((data["atom_pos"] - result["atom_pos"]) ** 2).sum(axis=-1)
            dist2 = jnp.clip(dist2, 0, do_clip[:, None])
            atom_mask = data["atom_mask"]
            if c.x_late:
                atom_mask *= late_mask[..., None]
            x_loss = (dist2 * atom_mask).sum(axis=-1) / 3
            x_loss /= jnp.maximum(atom_mask.sum(axis=-1), 1)
            x_loss = (x_loss * base_weight).sum()
            losses["x"] = x_loss
            total += c.x_weight * x_loss
            if c.x_trajectory and not c.no_trajectory_loss:
                dist2 = ((data["atom_pos"][None] - result["atom_trajectory"]) ** 2).sum(axis=-1)
                dist2 = jnp.clip(dist2, 0, do_clip[None, :, None])
                xt_loss = (dist2 * data["atom_mask"][None]).sum(axis=-1) / 3
                xt_loss /= jnp.maximum(data["atom_mask"].sum(axis=-1), 1)
                xt_loss = (xt_loss * base_weight).sum(axis=-1)
                xt_loss = xt_loss.mean()
                losses["x_trajectory"] = xt_loss
                total += c.x_weight * xt_loss
        # diffusion (rotation-space)
        if c.rotation_weight:
            gt_frames, _ = extract_aa_frames(
                Vec3Array.from_array(data["atom_pos"]))
            frames, _ = extract_aa_frames(
                Vec3Array.from_array(result["atom_pos"]))
            rotation_product = (gt_frames.rotation.inverse() @ frames.rotation).to_array()
            rotation_loss = ((rotation_product - jnp.eye(rotation_product.shape[-1])) ** 2).sum(axis=(-1, -2))
            rotation_loss = (rotation_loss * base_weight).sum()
            losses["rotation"] = rotation_loss
            total += 2 * c.rotation_weight * rotation_loss
            if c.refine and not c.no_trajectory_loss:
                frames, _ = jax.vmap(extract_aa_frames)(
                    Vec3Array.from_array(result["trajectory"]))
                rotation_product = (gt_frames.rotation.inverse()[None] @ frames.rotation).to_array()
                rotation_loss = ((rotation_product - jnp.eye(rotation_product.shape[-1])) ** 2).sum(axis=(-1, -2))
                rotation_loss = (rotation_loss * base_weight[None]).sum(axis=-1)
                rotation_loss = rotation_loss.mean()
                losses["rotation_trajectory"] = rotation_loss
                total += c.rotation_weight * rotation_loss

        # diffusion (locally-aligned space)
        if c.local_weight:
            gt_neighbours = extract_neighbours(0, 8, 0)(
                Vec3Array.from_array(data["atom_pos"]),
                resi, chain, batch, mask)
            gt_pos_neighbours = Vec3Array.from_array(
                data["atom_pos"][gt_neighbours])
            pos_neighbours = Vec3Array.from_array(
                result["atom_pos"][gt_neighbours])
            mask_neighbours = (gt_neighbours != -1)[..., None] * data["atom_mask"][gt_neighbours]
            gt_local_positions = gt_frames[:, None, None].apply_inverse_to_point(
                gt_pos_neighbours)
            local_positions = frames[:, None, None].apply_inverse_to_point(
                pos_neighbours)
            dist2 = (local_positions - gt_local_positions).norm2()
            dist2 = jnp.clip(dist2, 0, do_clip[:, None, None])
            local_loss = (dist2 * mask_neighbours).sum(axis=(-1, -2)) / 3
            local_loss /= jnp.maximum(mask_neighbours.astype(jnp.float32).sum(axis=(-1, -2)), 1)
            local_loss = (local_loss * base_weight).sum()
            losses["local"] = local_loss
            total += c.local_weight * local_loss
        # local / rotamer loss FIXME
        frames, local_positions = jax.vmap(extract_aa_frames)(
            Vec3Array.from_array(result["trajectory"]))
        gt_local_positions = gt_frames[:, None].apply_inverse_to_point(
            Vec3Array.from_array(data["pos"]))
        dist2 = (local_positions - gt_local_positions[None]).norm2()
        dist2 = jnp.clip(dist2, 0, do_clip[None, :, None])
        local_loss = (dist2 * late_mask[None, :, None]).mean(axis=-1) / 3
        local_loss = (local_loss * base_weight).sum(axis=-1)
        losses["local"] = 2 * local_loss[-1] + local_loss.mean()
        total += 1.0 * local_loss
        frames = frames[-1]
        gt_local_positions = gt_frames[:, None].apply_inverse_to_point(
            Vec3Array.from_array(data["atom_pos"]))
        local_positions = frames[:, None].apply_inverse_to_point(
            Vec3Array.from_array(result["atom_pos"]))
        dist2 = (local_positions - gt_local_positions).norm2()
        dist2 = jnp.clip(dist2, 0, do_clip[:, None])
        rotamer = (dist2 * late_mask[:, None] * data["atom_mask"]).sum(axis=-1) / 3
        rotamer /= jnp.maximum(data["atom_mask"].sum(axis=-1), 1)
        rotamer = (rotamer * base_weight).sum()
        losses["rotamer"] = rotamer
        total += 10.0 * 3 * rotamer
        
        # violation loss
        if c.violation_weight:
            res_mask = data["mask"]
            if c.diffusion_kind == "edm":
                res_mask *= data["t_pos"][:, 0] < 2.0
            else:
                res_mask *= data["t_pos"][:, 0] < 0.5
            pred_mask = get_atom14_mask(data["aa_gt"]) * res_mask[:, None]
            # FIXME: ensure our data is in atom14 format
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
        # FAPE
        if c.fape_weight:
            fape_loss, *_ = all_atom_fape(
                Vec3Array.from_array(result["atom_pos"]),
                Vec3Array.from_array(data["atom_pos"]), data["aa_gt"],
                batch, data["chain_index"],
                data["atom_mask"]
            )
            losses["fape"] = fape_loss
            total += 2 * c.fape_weight * fape_loss
        if c.fape_weight and c.refine and not c.no_trajectory_loss:
            trajectory_fape, *_ = jax.vmap(partial(all_atom_fape, sidechain=False),
                                     in_axes=(0, None, None, None, None, None))(
                Vec3Array.from_array(result["trajectory"]),
                Vec3Array.from_array(data["pos"]), data["aa_gt"],
                batch, data["chain_index"],
                data["atom_mask"]
            )
            trajectory_fape_loss = (trajectory_fape * trajectory_weight).sum()
            losses["trajectory_fape"] = trajectory_fape_loss
            total += c.fape_weight * trajectory_fape_loss
        # FIXME: monitor radius error
        def index_radius(ca, index, mask):
            center = index_mean(ca, index, mask[..., None])
            radius = jnp.sqrt(jnp.maximum(index_mean(((ca - center) ** 2).sum(axis=-1), index, mask), 1e-6))
            return center, radius
        gt_ca = data["atom_pos"][:, 1]
        ca = result["trajectory"][:, :, 1]
        _, gt_chain_radius = index_radius(gt_ca, data["chain_index"], mask)
        _, gt_batch_radius = index_radius(gt_ca, data["batch_index"], mask)
        _, chain_radius = jax.vmap(index_radius, in_axes=(0, None, None))(ca, data["chain_index"], mask)
        _, batch_radius = jax.vmap(index_radius, in_axes=(0, None, None))(ca, data["batch_index"], mask)
        chain_radius_loss = (base_weight * mask * (chain_radius - gt_chain_radius) ** 2).mean(axis=0).sum()
        batch_radius_loss = (base_weight * mask * (batch_radius - gt_batch_radius) ** 2).mean(axis=0).sum()
        losses["chain_radius"] = chain_radius_loss
        losses["batch_radius"] = batch_radius_loss
        losses["radius"] = chain_radius_loss + batch_radius_loss
        return total, losses

def get_angle_positions(aa_gt, local, pos):
    # FIXME: actually use aa_gt features & pos features
    frames, local_positions = extract_aa_frames(Vec3Array.from_array(pos))
    directions = local_positions.normalized()
    features = [
        local_positions.to_array().reshape(local_positions.shape[0], -1),
        distance_rbf(local_positions.norm(),
                     0.0, 10.0, 16).reshape(local_positions.shape[0], -1),
        # distance_rbf(directions[:, :, None].dot(local_positions[:, None, :]),
        #              -5.0, 5.0, 16).reshape(local_positions.shape[0], -1),
        jax.nn.one_hot(aa_gt, 21, axis=-1)
    ]
    raw_angles = MLP(
        local.shape[-1] * 2, 7 * 2, bias=False,
        activation=jax.nn.gelu, final_init="linear")(
            jnp.concatenate(features, axis=-1))

    # raw_angles = Linear(7 * 2, bias=False, initializer=init_glorot())(local)
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

    def __call__(self, aa, local, pos, resi, chain, batch, mask,
                 priority=None):
        c = self.config
        if c.aa_decoder_kind == "adm":
            local += Linear(
                local.shape[-1],
                initializer=init_glorot())(jax.nn.one_hot(aa, 21))
        neighbours = extract_neighbours(num_index=0, num_spatial=32, num_random=0)(
            Vec3Array.from_array(pos), resi, chain, batch, mask)
        local = AADecoderStack(c, depth=c.aa_decoder_depth)(
            aa, local, pos, neighbours,
            resi, chain, batch, mask,
            priority=priority)
        local = hk.LayerNorm([-1], True, True)(local)
        return Linear(20, initializer=init_zeros(), bias=False)(local), local

    def sample(self, local, pos, resi, chain, batch, mask, priority=None):
        c = self.config
        if c.aa_decoder_kind == "adm":
            aa_init = 20 * jnp.ones_like(resi)
            def adm_body(i, aa):
                logits, _ = self(aa, local, pos, resi, chain, batch, mask)
                logits = jax.nn.log_softmax(logits, axis=-1)
                logits /= c.temperature if c.temperature else 0.1
                p = jax.nn.softmax(logits, axis=-1)
                h = (p * logits).sum(axis=-1)
                h = jnp.where(aa == 2, jnp.inf, h)
                min_h_index = jnp.argmin(h, axis=0)
                aa = aa.at[min_h_index].set(
                    jax.random.categorical(hk.next_rng_key(), logits[min_h_index], axis=-1))
                return aa
            aa = hk.fori_loop(0, resi.shape[0], adm_body, aa_init)
        else:
            aa_init = 20 * jnp.ones_like(resi)
            logits = self(aa, local, pos, resi, chain, batch, mask)
            aa = jnp.argmax(logits, axis=-1)
        return aa

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
        # otherwise, independently predict logits for each
        # sequence position
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

    def __call__(self, local, pos, initial_pos,
                 condition, time,
                 resi, chain, batch, mask,
                 pair_condition=None,
                 cyclic_mask=None):
        c = self.config
        # if we want to run on a fixed set of neighbours
        # per diffusion step:
        fixed_neighbours = extract_neighbours(
            num_index=c.index_neighbours,
            num_spatial=c.spatial_neighbours,
            num_random=c.random_neighbours)(
                Vec3Array.from_array(pos),
                resi, chain, batch, mask
            )
        if c.pair_condition:
            fixed_neighbours = get_contact_neighbours(c.condition_neighbours or 16)(
                pair_condition, mask, fixed_neighbours)
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                if c.recompute_neighbours:
                    # if we want to re-compute neighbours
                    # at each diffusion step, extract them
                    # once for each block
                    neighbours = extract_neighbours(
                        num_index=c.index_neighbours,
                        num_spatial=c.spatial_neighbours,
                        num_random=c.random_neighbours)(
                            Vec3Array.from_array(pos),
                            resi, chain, batch, mask)
                    if c.pair_condition:
                        neighbours = get_contact_neighbours(c.condition_neighbours or 16)(
                            pair_condition, mask, neighbours)
                else:
                    # otherwise use the precomputed
                    # neighbours from above
                    neighbours = fixed_neighbours
                # run the diffusion block
                features, pos = block(c)(
                    features, pos, initial_pos,
                    condition, time, neighbours,
                    resi, chain, batch, mask,
                    pair_condition=pair_condition,
                    cyclic_mask=cyclic_mask)
                trajectory_output = pos
                if c.return_feature_trajectory:
                    trajectory_output = (pos, features)
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
        return local, pos, trajectory

class VPDiffusionBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, initial_pos,
                 condition, time, neighbours,
                 resi, chain, batch, mask,
                 pair_condition=None,
                 cyclic_mask=None):
        c = self.config
        residual_update, residual_input, local_shape = get_residual_gadgets(
            features, c.resi_dual)
        # _, initial_local_pos = extract_aa_frames(
        #     Vec3Array.from_array(initial_pos))
        _, current_local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        # add condition information to local features
        features = residual_update(
            features,
            Linear(local_shape, initializer=init_glorot(), bias=False)(
                condition))
        # add local position features to local features
        features = residual_update(
            features,
            PositionToLocal(c.local_size)(
                current_local_pos.to_array()))
        # global update of local features
        features = residual_update(
            features,
            GlobalUpdate(c)(
                residual_input(features),
                chain, batch, mask))
        # sparse structure message or attention
        # choose if we want to use attention
        attention_module = SparseStructureAttention
        pair_mask = make_pair_mask(mask, neighbours)
        # pair = Linear(c.pair_size, bias=False)(
        #     distance_features(Vec3Array.from_array(initial_pos), neighbours, d_max=22.0))
        pair = Linear(c.pair_size, bias=False)(
            distance_features(Vec3Array.from_array(pos), neighbours, d_max=22.0))
        # pair += Linear(c.pair_size, bias=False)(
        #     rotation_features(extract_aa_frames(Vec3Array.from_array(initial_pos))[0], neighbours))
        pair += Linear(c.pair_size, bias=False)(
            rotation_features(extract_aa_frames(Vec3Array.from_array(pos))[0], neighbours))
        if not c.no_relative_position:
            pair += Linear(c.pair_size, bias=False)(
                sequence_relative_position(32, one_hot=True, cyclic=c.cyclic)(
                    resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
        if pair_condition is not None:
            # index = jnp.arange(pair.shape[0], dtype=jnp.int32)
            # pair_condition = pair_condition[index[:, None], neighbours]
            pair_condition = jnp.take_along_axis(pair_condition, neighbours[:, :, None], axis=1)
            pair_condition = jnp.where(pair_mask[..., None], pair_condition, 0)
            pair += Linear(c.pair_size, bias=False)(pair_condition)
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = jnp.where(pair_mask[..., None], pair, 0)
        # attention / message passing module
        features = residual_update(
            features,
            attention_module(c)(
                residual_input(features), pos / c.sigma_data, pair, pair_mask,
                neighbours, resi, chain, batch, mask))
        # local feature transition (always enabled)
        features = residual_update(
            features,
            GatedMLP(local_shape * c.factor, local_shape, activation=jax.nn.gelu,
                     final_init=init_zeros())(residual_input(features)))
        if c.resi_dual:
            local, incremental = features
            local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local_norm = residual_input(features)
        # update positions
        if c.refine:
            split_scale_out = c.sigma_data
            pos = update_positions(pos, local_norm,
                                   scale=split_scale_out,
                                   symm=c.symm) # FIXME: implement more general symmetrisation
        return features, pos.astype(local.dtype)

class EDMDiffusionBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, initial_pos,
                 condition, time, neighbours,
                 resi, chain, batch, mask,
                 pair_condition=None,
                 cyclic_mask=None):
        c = self.config
        residual_update, residual_input, local_shape = get_residual_gadgets(
            features, c.resi_dual)
        # set up model preconditioning
        scale_factors = preconditioning_scale_factors(
            c, time, c.sigma_data)
        current_pos = pos
        _, current_local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        # if we're using initial positions
        # NOTE: if you notice a discrepancy in radius losses
        # this is probably where it's coming from.
        # Scale to center requires the model to learn where to
        # push chains - they are no longer pushed to the
        # center of mass of the entire complex anymore.
        if c.scale_to_chain_center:
            initial_pos = scale_to_center(
                initial_pos, time,
                scale_factors["in"],
                chain, mask)
            scaled_current_pos = scale_to_center(
                current_pos, time,
                scale_factors["refine"],
                chain, mask)
        else:
            initial_pos *= scale_factors["in"][..., None]
            scaled_current_pos = current_pos * scale_factors["refine"]
        _, initial_local_pos = extract_aa_frames(
            Vec3Array.from_array(initial_pos))
        # add condition information to local features
        features = residual_update(
            features,
            Linear(local_shape, initializer=init_glorot(), bias=False)(
                condition))
        # add local position features to local features
        features = residual_update(
            features,
            PositionToLocal(c.local_size)(
                initial_local_pos.to_array()))
        # global update of local features
        features = residual_update(
            features,
            GlobalUpdate(c)(
                residual_input(features),
                chain, batch, mask))
        # sparse structure message or attention
        # choose if we want to use attention
        attention_module = SparseStructureAttention
        # NOTE: Michael, don't be stupid and scale the positions
        # again, which you already scale 5 lines earlier!!!!!
        # embed pair features
        pair_mask = make_pair_mask(mask, neighbours)
        pair = Linear(c.pair_size, bias=False)(
            distance_features(Vec3Array.from_array(initial_pos), neighbours, d_max=10.0))
        pair += Linear(c.pair_size, bias=False)(
            distance_features(Vec3Array.from_array(scaled_current_pos), neighbours, d_max=10.0))
        pair += Linear(c.pair_size, bias=False)(
            rotation_features(extract_aa_frames(Vec3Array.from_array(initial_pos))[0], neighbours))
        pair += Linear(c.pair_size, bias=False)(
            rotation_features(extract_aa_frames(Vec3Array.from_array(current_pos))[0], neighbours))
        if not c.no_relative_position:
            pair += Linear(c.pair_size, bias=False)(
                sequence_relative_position(32, one_hot=True, cyclic=c.cyclic)(
                    resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
        if pair_condition is not None:
            index = jnp.arange(pair.shape[0], dtype=jnp.int32)
            pair_condition = pair_condition[index[:, None], neighbours]
            pair_condition = jnp.where(pair_mask[..., None], pair_condition, 0)
            pair += Linear(c.pair_size, bias=False)(pair_condition)
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = jnp.where(pair_mask[..., None], pair, 0)
        # attention / message passing module
        features = residual_update(
            features,
            attention_module(c)(
                residual_input(features), scaled_current_pos, pair, pair_mask,
                neighbours, resi, chain, batch, mask))
        # local feature transition (always enabled)
        features = residual_update(
            features,
            GatedMLP(local_shape * c.factor, local_shape, activation=jax.nn.gelu,
                     final_init=init_zeros())(residual_input(features)))
        if c.resi_dual:
            local, incremental = features
            local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local_norm = residual_input(features)
        # update positions
        if c.refine:
            split_scale_out = scale_factors["out"]
            pos = update_positions(pos, local_norm,
                                   scale=split_scale_out,
                                   symm=c.symm)
        return features, pos.astype(local.dtype) 

class DiffusionBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos,
                 initial_pos, condition, time,
                 neighbours, resi, chain, batch, mask,
                 pair_condition=None):
        c = self.config
        residual_update, residual_input, local_shape = get_residual_gadgets(
            features, c.resi_dual)
        # set up model preconditioning
        scale_factors = preconditioning_scale_factors(
            c, time, c.sigma_data)
        current_pos = scale_factors["refine"] * pos
        _, current_local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        # FIXME: use current_local_pos as well
        local_pos_features = [(current_local_pos / scale_factors["refine"]).to_array()]
        # if we're using initial positions
        if initial_pos is not None:
            # FIXME: scale to center?
            if c.preconditioning in ("edm", "vpedm"):
                initial_pos = scale_to_center(
                    initial_pos, time,
                    scale_factors["in"],
                    chain, mask)
            else:
                initial_pos = scale_factors["in"][..., None] * initial_pos
            _, initial_local_pos = extract_aa_frames(
                Vec3Array.from_array(initial_pos))
            local_pos_features.append(initial_local_pos.to_array())
        local_pos_features = jnp.concatenate(local_pos_features, axis=-2)
        # add condition information to local features
        features = residual_update(
            features,
            Linear(local_shape, initializer=init_glorot(), bias=False)(
                condition))
        # add local position features to local features
        features = residual_update(
            features,
            PositionToLocal(c.local_size)(
                local_pos_features))
        # global update of local features
        if c.global_update:
            features = residual_update(
                features,
                GlobalUpdate(c)(
                    residual_input(features),
                    chain, batch, mask))
        # sparse structure message or attention
        # choose if we want to use attention
        attention_module = (
            SparseStructureAttention
            if c.use_attention
            else SparseStructureMessage
        )
        # embed pair features
        d_max = jnp.where(scale_factors["refine"] == 1, 22.0, 10.0)
        # FIXME: sum pair embedding
        pair, pair_mask = sum_edm_pair_embedding(
            c, use_local=not c.use_attention, d_max=d_max)(
            residual_input(features),
            [Vec3Array.from_array(current_pos), Vec3Array.from_array(initial_pos)],
            pair_condition, neighbours, resi, chain, batch, mask
        )
        # attention / message passing module
        features = residual_update(
            features,
            attention_module(c)(
                residual_input(features), current_pos, pair, pair_mask,
                neighbours, resi, chain, batch, mask))
        # local feature transition (always enabled)
        features = residual_update(
            features,
            # FIXME: GatedMLP init zeros
            MLP(local_shape * c.factor, local_shape, activation=jax.nn.gelu,
                bias=False, final_init=init_zeros())(residual_input(features)))
        if c.resi_dual:
            local, incremental = features
            local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local_norm = residual_input(features)
        # update positions
        if c.refine:
            pos = update_positions(pos, local_norm,
                                   scale=scale_factors["out"])
        return features, pos.astype(local.dtype) 

class RecomputedDiffusionBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, offset,
                 initial_offset, condition, sigma,
                 neighbours, resi, chain, batch, mask,
                 pair_condition=None):
        c = self.config
        local, incremental = features
        frames, local_offset = extract_aa_frames(Vec3Array.from_array(offset))
        sigma_data = 10.0 # FIXME: standard deviation of PDB
        in_scale = 1 / jnp.sqrt(sigma_data ** 2 + sigma ** 2)
        out_scale = sigma * sigma_data * in_scale
        skip_scale = sigma_data ** 2 * in_scale ** 2
        current_in_scale = 1 / sigma_data
        in_scaled_offset = in_scale[..., None] * initial_offset
        _, in_scaled_local_offset = extract_aa_frames(Vec3Array.from_array(in_scaled_offset))
        current_in_scaled_offset = current_in_scale * offset
        # neighbours = extract_neighbours(c)(Vec3Array.from_array(offset), resi, chain, batch, mask)
        local, incremental = resi_dual(
            local, incremental,
            Linear(local.shape[-1], initializer=init_glorot())(condition))
        local, incremental = resi_dual(
            local, incremental,
            PositionToLocal(c.local_size)(
                # only initial pos in here,
                # vs initial pos & current pos in rewrite
                in_scaled_local_offset.to_array()))
        local, incremental = resi_dual(
            local, incremental,
            GlobalUpdate(c)(
                local, chain, batch, mask))
        # this here is the most different part
        # of the module.
        # so something could be going wrong in the pair embedding
        # or in the attention block proper. I don't think it's the
        # attention block, as this is exactly the same in the rewrite
        local, incremental = resi_dual(
            local, incremental,
            DualSparseStructureAttention(c, d_max=10.0)(
                local, current_in_scaled_offset, in_scaled_offset, resi, chain, batch,
                neighbours, mask, pair_condition=pair_condition))
        # this is a relu vs a gelu in the rewrite, why?
        # however, the issue was present before this was changed
        # to a gelu MLP in the rewrite as well
        local, incremental = resi_dual(
            local, incremental,
            MLP(local.shape[-1] * c.factor,
                local.shape[-1], final_init=init_glorot())(local))
        
        # this is the exact same in the rewrite
        local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        offset_update = out_scale * Linear(offset.shape[-2] * 3, initializer=init_zeros())(local_norm)
        offset_update = Vec3Array.from_array(offset_update.reshape(*offset_update.shape[:-1], -1, 3))
        local_offset += offset_update
        # project updated offset to global coordinates
        offset = frames[..., None].apply_to_point(local_offset).to_array()
        return (local, incremental), offset.astype(local.dtype)

class DualSparseStructureAttention(hk.Module):
    def __init__(self, config, d_min=0.0, d_max=10.0, name: Optional[str] = None):
        super().__init__(name)
        self.config = config
        self.d_min = d_min
        self.d_max = d_max

    def __call__(self, local, pos, prev_pos, resi, chain, batch, neighbours, mask, pair_condition=None):
        c = self.config
        pos = Vec3Array.from_array(pos.astype(jnp.float32))
        frames, local_pos = extract_aa_frames(pos)
        pair = 0.0
        pair_mask = mask[:, None] * mask[neighbours] * (neighbours != -1)
        # current embedding:
        dist = (pos[:, None, :4, None] - pos[neighbours, None, :4]).norm()
        dist = dist.reshape(*dist.shape[:-2], -1)
        pair += Linear(64, bias=False)(distance_rbf(dist, 0.0, self.d_max, 16).reshape(*dist.shape[:-1], -1))
        rot = frames[:, None].inverse().rotation @ frames[neighbours].rotation
        rot = rot.to_array().reshape(*rot.shape, -1)
        pair += Linear(64, bias=False)(rot)
        if prev_pos is not None:
            prev_pos = Vec3Array.from_array(prev_pos)
            prev_frames, _ = extract_aa_frames(prev_pos)
            prev_dist = (prev_pos[:, None, :4, None] - prev_pos[neighbours, None, :4]).norm()
            prev_dist = prev_dist.reshape(*prev_dist.shape[:-2], -1)
            pair += Linear(64, bias=False)(distance_rbf(prev_dist, 0.0, self.d_max, 16).reshape(*prev_dist.shape[:-1], -1))
            prev_rot = prev_frames[:, None].inverse().rotation @ prev_frames[neighbours].rotation
            prev_rot = prev_rot.to_array().reshape(*prev_rot.shape, -1)
            pair += Linear(64, bias=False)(prev_rot)

        difference = jnp.clip(resi[:, None] - resi[neighbours], -32, 32) + 32
        other_chain = (chain[:, None] != chain[neighbours]) * (batch[:, None] == batch[neighbours]) * (neighbours != -1)
        difference = jnp.where(other_chain, 65, difference)
        relative = jax.nn.one_hot(difference, 66, axis=-1)
        pair += Linear(64, bias=False)(relative)
        pair = hk.LayerNorm([-1], True, True)(pair)
        local_update = SparseInvariantPointAttention(
            heads=c.heads, size=c.key_size, normalize=False)(
            local, pair, frames.to_array(), neighbours, pair_mask)
        return local_update

def get_residual_gadgets(features, use_resi_dual=True):
    residual_update = (lambda x, y: resi_dual(*x, y)) if use_resi_dual else prenorm_skip
    residual_input = resi_dual_input if use_resi_dual else prenorm_input
    local_shape = features[0].shape[-1] if use_resi_dual else features.shape[-1]
    return residual_update, residual_input, local_shape

def pos_pair_features(config, d_max=22.0, position_scale=0.1):
    c = config
    def inner(pos, neighbours, resi, chain, batch, mask):
        frames, _ = extract_aa_frames(pos)
        features = Linear(c.pair_size, bias=False)(
            frame_pair_features(
                extract_aa_frames(pos)[0],
                pos, neighbours, d_max=d_max))
        if c.pair_vector_features:
            features += Linear(c.pair_size, bias=False)(
                frame_pair_vector_features(
                    frames, pos, neighbours,
                    position_scale))
        features += Linear(c.pair_size, bias=False)(
            sequence_relative_position(
                c.relative_position_encoding_max, one_hot=True,
                cyclic=c.cyclic, identify_ends=c.identify_ends)(
                    resi, chain, batch, neighbours=neighbours))
        return features
    return inner

def make_pair_mask(mask, neighbours):
    return mask[:, None] * mask[neighbours] * (neighbours != -1)

def sum_edm_pair_embedding(config, use_local=False, d_max=22.0):
    c = config
    def inner(local, current_pos, initial_pos, pair_condition,
              neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours] * (neighbours != -1)
        frames, _ = extract_aa_frames(current_pos)
        features = Linear(c.pair_size, bias=False)(
            frame_pair_features(
                extract_aa_frames(current_pos)[0],
                current_pos, neighbours, d_max=d_max))
        if initial_pos is not None:
            features += Linear(c.pair_size, bias=False)(
                frame_pair_features(
                    extract_aa_frames(
                        initial_pos)[0], initial_pos,
                        neighbours, d_max=d_max))
        if c.pair_vector_features:
            features += Linear(c.pair_size, bias=False)(
                frame_pair_vector_features(
                    frames, current_pos, neighbours,
                    1 / c.position_scale))
        features += Linear(c.pair_size, bias=False)(
            sequence_relative_position(
                c.relative_position_encoding_max, one_hot=True,
                cyclic=c.cyclic, identify_ends=c.identify_ends)(
                    resi, chain, batch, neighbours=neighbours))
        if use_local:
            local_features = Linear(
                c.pair_size, bias=False,
                initializer=init_glorot())(local)[:, None]
            local_features += Linear(
                c.pair_size, bias=False,
                initializer=init_glorot())(local)[neighbours]
            features += local_features
        if pair_condition is not None:
            idb = jnp.arange(pair_condition.shape[0], dtype=jnp.int32)
            pair_condition = pair_condition[idb[:, None], neighbours]
            features += Linear(c.pair_size, bias=False)(pair_condition)
        features = hk.LayerNorm([-1], True, True)(features)
        features = jnp.where(pair_mask[..., None], features, 0)
        return features, pair_mask
    return inner

def edm_pair_embedding(config, use_local=False, d_max=22.0):
    c = config
    def inner(local, current_pos, initial_pos, pair_condition,
              neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours] * (neighbours != -1)
        frames, _ = extract_aa_frames(current_pos)
        features = [
            frame_pair_features(
                extract_aa_frames(current_pos)[0], current_pos, neighbours, d_max=d_max)
        ]
        if initial_pos is not None:
            features.append(frame_pair_features(
                extract_aa_frames(initial_pos)[0], initial_pos, neighbours, d_max=d_max))
        if c.pair_vector_features:
            features.append(frame_pair_vector_features(
                    frames, current_pos, neighbours,
                    1 / c.position_scale))
        features.append(sequence_relative_position(
            c.relative_position_encoding_max, one_hot=True,
            cyclic=c.cyclic, identify_ends=c.identify_ends)(
                resi, chain, batch, neighbours=neighbours
            ))
        if use_local:
            local_features = Linear(
                c.pair_size, bias=True,
                initializer=init_glorot())(local)[:, None]
            local_features += Linear(
                c.pair_size, bias=True,
                initializer=init_glorot())(local)[neighbours]
            features.append(local_features)
        if pair_condition is not None:
            idb = jnp.arange(pair_condition.shape[0], dtype=jnp.int32)
            pair_condition = pair_condition[idb[:, None], neighbours]
            features.append(pair_condition)
        features = jnp.concatenate(features, axis=-1)
        if c.sum_features:
            tmp = 0.0
            for data in features:
                tmp += Linear(c.pair_size, initializer=init_glorot(), bias=False)(data)
            features = tmp
        else:
            features = Linear(c.pair_size, initializer=init_glorot())(features)
        features = hk.LayerNorm([-1], True, True)(features)
        features = jnp.where(pair_mask[..., None], features, 0)
        return features, pair_mask
    return inner

class Condition(hk.Module):
    def __init__(self, config, name: Optional[str] = "condition"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, dssp, pos, pos_mask, residue_index, chain_index, batch_index, do_condition=None):
        sequence_mask, dssp_mask, structure_mask, dssp_mean_mask = self.get_masks(
            chain_index, batch_index, do_condition=do_condition)

        structure_mask = structure_mask * pos_mask.any(axis=-1)
        pos_mask = jnp.where(sequence_mask[..., None], pos_mask, pos_mask.at[:, 5:].set(0))

        c = self.config
        aa_latent = Linear(c.local_size, initializer=init_glorot())(jax.nn.one_hot(aa, 21, axis=-1))
        aa_latent = hk.LayerNorm([-1], True, True)(aa_latent)

        dssp_latent = Linear(c.local_size, initializer=init_glorot())(jax.nn.one_hot(dssp, 3, axis=-1))
        dssp_latent = hk.LayerNorm([-1], True, True)(dssp_latent)

        dssp_mean = index_mean(jax.nn.one_hot(dssp, 3, axis=-1), chain_index, pos_mask.any(axis=-1, keepdims=True))
        dssp_mean /= jnp.maximum(dssp_mean.sum(axis=-1, keepdims=True), 1e-6)
        if (do_condition is not None) and (do_condition["dssp_mean"] is not None):
            dssp_mean = do_condition["dssp_mean"]
        dssp_mean = Linear(c.local_size, initializer=init_glorot())(dssp_mean)
        dssp_mean = hk.LayerNorm([-1], True, True)(dssp_mean)

        pos = jnp.where(pos_mask[..., None], pos, pos[..., 1:2, :])
        frames, local_positions = extract_aa_frames(
            Vec3Array.from_array(pos.astype(jnp.float32)))

        # FIXME: let's not make ourselves non-equivariant
        # pos = jnp.where(structure_mask[..., None, None], pos, 0)
        offsets = jnp.arange(0, pos.shape[0])[:, None] + jnp.linspace(-2, 2, 5, dtype=jnp.int32)[None, :]
        invalid = (offsets < 0) * (batch_index[offsets] != batch_index[:, None]) * (chain_index[offsets] != chain_index[:, None])
        # FIXME: explicitly mask bad positions
        invalid *= structure_mask[offsets] < 1
        local_positions = pos[offsets]
        local_positions = Vec3Array.from_array(local_positions)
        local_mask = pos_mask[offsets]
        local_positions = frames[:, None, None].apply_inverse_to_point(
            local_positions).to_array()
        local_positions = jnp.where(
            local_mask[..., None], local_positions, 0)
        # set invalid positions equal to 0
        local_positions = jnp.where(
            invalid[:, :, None, None], 0, local_positions)
        local_positions = local_positions[..., :4, :]
        local_positions = local_positions.reshape(
            local_positions.shape[0], -1, 3)
        distance_embedding = distance_rbf(
            jnp.sqrt((local_positions ** 2).sum(axis=-1)), 0, 22, 64).reshape(*local_positions.shape[:-2], -1)
        direction = local_positions / jnp.sqrt(jnp.maximum((local_positions ** 2).sum(axis=-1, keepdims=True), 1e-6))
        direction = direction.reshape(*local_positions.shape[:-2], -1)
        local = Linear(c.local_size, bias=False)(direction) \
              + Linear(c.local_size, bias=False)(distance_embedding)
        local = hk.LayerNorm([-1], False, False)(local)
        neighbours = extract_neighbours(
            num_index=16, num_spatial=16, num_random=32)(
                Vec3Array.from_array(pos),
                residue_index, chain_index, batch_index, structure_mask)
        local = EncoderStack(c, depth=2)(
            local, pos, neighbours,
            residue_index, chain_index, batch_index,
            structure_mask)
        bb_latent = local

        condition = jnp.where(sequence_mask[..., None], aa_latent, 0) \
                  + jnp.where(dssp_mask[..., None], dssp_latent, 0) \
                  + jnp.where(structure_mask[..., None], bb_latent, 0) \
                  + jnp.where(dssp_mean_mask[..., None], dssp_mean, 0)
        return condition, sequence_mask

    def get_masks(self, chain_index, batch_index, do_condition=None):
        if do_condition is not None:
            sequence_mask = do_condition["aa"] != 20
            sse_mask = do_condition["dssp"] != 3
            structure_mask = do_condition["structure"]["mask"]
            sse_mean_mask = do_condition["dssp_mean_mask"]
            return sequence_mask, sse_mask, structure_mask, sse_mean_mask
        mask_full = jnp.ones(batch_index.shape, dtype=jnp.bool_)
        mask_80 = jax.random.bernoulli(hk.next_rng_key(), p=0.8, shape=batch_index.shape)
        mask_20 = ~mask_80
        mask_choices = jnp.stack((
            mask_full,
            mask_80,
            mask_20
        ), axis=-1)
        index = jnp.arange(batch_index.shape[0])
        mask_index = jax.random.randint(hk.next_rng_key(), batch_index.shape, 0, 3)[batch_index]
        mask = mask_choices[index, mask_index]
        sequence_mask = mask * jax.random.bernoulli(hk.next_rng_key(), p=0.5, shape=batch_index.shape)[batch_index]
        sse_mask = mask * jax.random.bernoulli(hk.next_rng_key(), p=0.5, shape=batch_index.shape)[batch_index]
        structure_mask = mask * jax.random.bernoulli(hk.next_rng_key(), p=0.5, shape=batch_index.shape)[batch_index]
        sse_mean_mask = jax.random.bernoulli(hk.next_rng_key(), p=0.5, shape=batch_index.shape)[chain_index]
        return sequence_mask, sse_mask, structure_mask, sse_mean_mask

def get_pair_condition(pos, aa, block_adjacency, pos_mask, resi, chain, batch, do_condition=None):
    same_batch = batch[:, None] == batch[None, :]
    same_chain = (chain[:, None] == chain[None, :]) * same_batch
    p = jax.random.uniform(hk.next_rng_key(), (batch.shape[0],), minval=0.0, maxval=0.1)[batch]
    noise = jax.random.bernoulli(hk.next_rng_key(), p)
    results = interactions(pos, aa, block_adjacency, pos_mask, resi, chain, batch, block_noise=noise)
    pair_condition = jnp.stack((
        results["chain_contact"],
        results["relative_hotspot"],
        results["relative_hotspot"].T,
        results["block_contact"]), axis=-1)
    # 50% chance to drop basic contact info for any chain
    drop_chain_contact_info = jax.random.bernoulli(hk.next_rng_key(), 0.5, (chain.shape[0],))[chain]
    # 20 - 80% chance to drop hotspot info for any residue in a chain
    drop_chain_hotspot_info = jax.random.bernoulli(hk.next_rng_key(), 0.5, (chain.shape[0],))[chain]
    # 0 - 50% chance to drop each individual hotspot
    p_drop = jax.random.uniform(hk.next_rng_key(), (chain.shape[0],), minval=0.0, maxval=0.5)[chain]
    drop_hotspot_info = jax.random.bernoulli(hk.next_rng_key(), p_drop)
    # 0 - 80% chance to drop block-contact info for any pair of residues
    p_drop = jax.random.uniform(hk.next_rng_key(), (chain.shape[0],), minval=0.0, maxval=0.8)[chain]
    drop_block_contact_info = jax.random.bernoulli(hk.next_rng_key(), p_drop)
    chain_contact_mask = drop_chain_contact_info[:, None] * drop_chain_contact_info[None, :]
    relative_hotspot_mask = drop_chain_hotspot_info[:, None] * drop_hotspot_info[None, :]
    relative_hotspot_mask = jnp.where(chain_contact_mask, relative_hotspot_mask, 0)
    block_contact_mask = drop_block_contact_info[:, None] * drop_block_contact_info[None, :]
    block_contact_mask = jnp.where(same_chain, block_contact_mask, 0)
    pair_mask = jnp.stack((
        chain_contact_mask,
        relative_hotspot_mask,
        relative_hotspot_mask.T,
        block_contact_mask
    ), axis=-1)
    return jnp.where(pair_mask, pair_condition, 0)

# TODO: move to utils
POLAR_THRESHOLD = 3.0
CONTACT_THRESHOLD = 8.0
def interactions(coords, aatype, block_adjacency, mask, resi, chain, batch, block_noise=None):
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

class PositionToLocal(hk.Module):
    def __init__(self, size, name: Optional[str] = "pos2local"):
        super().__init__(name)
        self.size = size

    def __call__(self, pos: jnp.ndarray):
        pos = pos - pos.mean(axis=-2, keepdims=True)
        pos /= jnp.sqrt(jnp.maximum((pos ** 2).sum(axis=-1, keepdims=True), 1e-12)).mean(axis=-2, keepdims=True)
        norm = jnp.sqrt(jnp.maximum((pos ** 2).sum(axis=-1), 1e-12))
        reshaped = pos.reshape(*pos.shape[:-2], -1)
        features = jnp.concatenate((reshaped, norm), axis=-1)
        hidden = Linear(4 * self.size, initializer=init_relu(), bias=False)(features)
        result = Linear(self.size, initializer=init_zeros(), bias=False)(jax.nn.gelu(hidden))
        return result

class GlobalUpdate(hk.Module):
    def __init__(self, factor, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.factor = factor

    def __call__(self, local, chain, batch, mask):
        local_update = Linear(local.shape[-1] * 2, initializer=init_relu())(local)
        # FIXME: gelu
        local_batch = jax.nn.relu(index_mean(local_update, batch, mask[..., None]))
        local_chain = jax.nn.relu(index_mean(local_update, chain, mask[..., None]))
        result = Linear(local.shape[-1], initializer=init_zeros())(
            local_batch + local_chain)
        return result

def update_positions(pos, local_norm, scale=10.0, symm=False):
    frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
    pos_update = scale * Linear(
        pos.shape[-2] * 3, initializer=init_zeros(),
        bias=False)(local_norm)
    # FIXME: implement more general symmetrisation
    if symm:
        mean_update = (pos_update[:100] + pos_update[100:200] + pos_update[200:300]) / 3
        pos_update = 0.5 * pos_update + 0.5 * jnp.concatenate((mean_update, mean_update, mean_update), axis=0)
    pos_update = Vec3Array.from_array(
        pos_update.reshape(*pos_update.shape[:-1], -1, 3))
    local_pos += pos_update

    # project updated pos to global coordinates
    pos = frames[..., None].apply_to_point(local_pos).to_array()
    return pos

def update_positions_convex(pos, local_norm, resi, chain, batch, mask,
                            scale=10.0, num_iterations=5, cyclic=False):
    frames, _ = extract_aa_frames(Vec3Array.from_array(pos))
    neighbours = extract_neighbours(
            num_index=16, num_spatial=16, num_random=32)(
                pos, resi, chain, batch, mask)
    pair_mask = mask[:, None] * mask[neighbours] * (neighbours != -1)
    relative_local_pos: Vec3Array = frames[neighbours, None].apply_inverse_to_point(
        Vec3Array.from_array(pos[:, None]))
    relative_distance = relative_local_pos.norm()
    relative_direction = relative_local_pos.normalized().to_array()
    pair = Linear(64, bias=False)(
        sequence_relative_position(32, one_hot=True, cyclic=cyclic)(
            resi, chain, batch, neighbours))
    pair += Linear(64, bias=False)(
        jnp.concatenate((
            relative_local_pos.to_array().reshape(*relative_local_pos.shape[:-1], -1) / scale,
            distance_rbf(relative_distance, bins=16).reshape(*relative_direction.shape[:2], -1),
            relative_direction.reshape(*relative_direction.shape[:2], -1)
        ), axis=-1))
    pair += Linear(64, bias=False)(local_norm[:, None])
    pair += Linear(64, bias=False)(local_norm[None, :])
    pair = hk.LayerNorm([-1], True, True)(pair)
    pair += MLP(128, 64, bias=False, activation=jax.nn.gelu, final_init=init_glorot())(pair)
    pair = hk.LayerNorm([-1], True, True)(pair)
    update = scale * Linear(
        relative_local_pos.shape[-1] * 3, bias=False,
        initializer=init_zeros())(pair)
    weight = Linear(
        relative_local_pos.shape[-1], bias=False,
        initializer=init_zeros())(pair)
    weight = jnp.where(pair_mask[..., None], weight, -1e9)
    weight = jax.nn.softmax(weight, axis=1)
    weight = jnp.where(pair_mask[..., None], weight, 0)
    update = update.reshape(*update.shape[:-1], relative_local_pos.shape[-1], 3)
    update = Vec3Array.from_array(update)
    relative_local_pos += update
    def iteration(i, current_pos):
        pos = Vec3Array.from_array(current_pos)
        frames, _ = extract_aa_frames(pos)
        global_pos = frames[neighbours, None].apply_to_point(relative_local_pos)
        current_pos = (global_pos.to_array() * weight[..., None] * pair_mask[..., None, None]).sum(axis=1)
        return current_pos
    pos = jax.lax.fori_loop(0, num_iterations, iteration, pos)
    return pos

def update_positions_consensus(pos, local_norm, resi, chain, batch, mask,
                               scale=10.0, num_iterations=10):
    frames, _ = extract_aa_frames(Vec3Array.from_array(pos))
    neighbours = extract_neighbours(
            num_index=16, num_spatial=16, num_random=32)(
                pos, resi, chain, batch, mask)
    pair_mask = mask[:, None] * mask[neighbours] * (neighbours != -1)
    relative_local_pos: Vec3Array = frames[neighbours, None].apply_inverse_to_point(
        Vec3Array.from_array(pos[:, None]))
    relative_distance = relative_local_pos.norm().to_array()
    relative_direction = relative_local_pos.normalized().to_array()
    pair = Linear(64, bias=False)(
        sequence_relative_position(32, one_hot=True)(
            resi, chain, batch, neighbours))
    pair += Linear(64, bias=False)(
        jnp.concatenate((
            distance_rbf(relative_distance, bins=16).reshape(*relative_direction.shape[:2], -1),
            relative_direction.reshape(*relative_direction.shape[:2], -1)
        ), axis=-1))
    pair += Linear(64, bias=False)(local_norm[:, None])
    pair += Linear(64, bias=False)(local_norm[None, :])
    output_data = MLP(128, 7 * pos.shape[-2], bias=False,
                      activation=jax.nn.gelu, final_init="zeros")(pair)
    old_new_weight = output_data[..., -1:]
    output_data = output_data[..., :-1]
    new_positions, new_position_weights = jnp.split(output_data, 2, axis=-1)
    new_positions = scale * new_position_weights.reshape(
        *new_position_weights.shape[:2], pos.shape[-2], 3)
    new_positions = old_new_weight[..., None] * relative_local_pos + (1 - old_new_weight)[..., None] * relative_local_pos.to_array()
    new_position_weights = new_position_weights.reshape(
        *new_position_weights.shape[:2], pos.shape[-2], 3)
    new_position_weights = jax.nn.gelu(new_position_weights)
    new_position_weights = jnp.where(pair_mask[..., None, None], new_position_weights, 0)
    def iteration(i, carry):
        mean_pos = carry
        frames, _ = extract_aa_frames(Vec3Array.from_array(mean_pos))
        global_positions = frames[neighbours, :].apply_to_point(Vec3Array.from_array(new_positions)).to_array()
        mean_pos = (new_position_weights * global_positions).sum(axis=1) / new_position_weights.sum(axis=1)
        return mean_pos
    pos = jax.lax.fori_loop(0, num_iterations, iteration, pos)
    return pos
