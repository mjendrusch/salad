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
    get_neighbours, distance_rbf, distance_one_hot, assign_sse,
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
    distance_features, direction_features, rotation_features
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
        seq, pos, dssp = self.encode(data)

        # set all-atom-position target
        atom_pos = pos_14
        atom_mask = atom_mask_14
        return dict(seq=seq, pos=pos, dssp=dssp,
                    chain_index=chain, mask=mask,
                    atom_pos=atom_pos, atom_mask=atom_mask,
                    all_atom_positions=pos_37,
                    all_atom_mask=atom_mask_37)

    def encode(self, data):
        pos = data["pos"]
        aatype = data["aa_gt"]
        # set up ncaco + pseudo cb positions
        backbone = positions_to_ncacocb(pos)
        pos = backbone
        seq = aatype
        dssp, _, _ = assign_dssp(pos, data["batch_index"], data["all_atom_mask"].any(axis=-1))
        return seq, pos, dssp

    def apply_diffusion(self, data):
        c = self.config
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        pos = Vec3Array.from_array(data["pos"])
        seq = data["seq"]
        if True:# c.focused_time:
            def _expit(x): return 1 / (1 + jnp.exp(-x))
            rand_pos = jax.random.normal(hk.next_rng_key(), (batch.shape[0], 1))[batch]
            t_pos = _expit(0.0 + 1.0 * rand_pos)
            rand_seq = jax.random.normal(hk.next_rng_key(), (batch.shape[0], 1))[batch]
            t_seq = _expit(0.0 + 1.0 * rand_seq)
        else:
            # FIXME: unreachable
            t_pos = jax.random.uniform(
                hk.next_rng_key(), (batch.shape[0], 1))[batch]
            t_seq = jax.random.uniform(
                hk.next_rng_key(), (batch.shape[0], 1))[batch]
        pos_noised = diffuse_coordinates_blend(
            hk.next_rng_key(), pos, mask, batch, t_pos, scale=c.sigma_data)
        seq_noised, corrupt_aa = diffuse_sequence(hk.next_rng_key(), seq, t_seq[:, 0])
        return dict(
            seq_noised=seq_noised, pos_noised=pos_noised.to_array(),
            t_pos=t_pos, t_seq=t_seq, corrupt_aa=corrupt_aa)

    def __call__(self, data):
        c = self.config
        diffusion = Diffusion(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))

        result = diffusion(data)
        total, losses = diffusion.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

# class StructureDiffusionInference(StructureDiffusion):
#     def __init__(self, config,
#                  name: Optional[str] = "structure_diffusion"):
#         super().__init__(config, name)
#         self.config = config

#     def __call__(self, data):
#         c = self.config
#         diffusion = Diffusion(c)

#         # prepare condition / task information
#         data.update(self.prepare_condition(data))
#         # apply noise to data
#         data.update(self.apply_diffusion(data))

#         result = diffusion(data, predict_aa=True)

#         # NOTE: also return noisy input for diagnostics
#         result["pos_input"] = data["pos_noised"]
        
#         return result

#     def prepare_condition(self, data):
#         c = self.config
#         # TODO: allow non-zero condition
#         result = dict(condition=jnp.zeros((data["pos"].shape[0], c.local_size)))
#         cond = Condition(c)
#         aa = jnp.zeros_like(data["aa_gt"])
#         aa_mask = jnp.zeros(aa.shape[0], dtype=jnp.bool_)
#         if "aa_condition" in data:
#             aa = data["aa_condition"]
#             aa_mask = aa != 20
#         dssp = jnp.zeros_like(data["aa_gt"])
#         dssp_mask = jnp.zeros(aa.shape[0], dtype=jnp.bool_)
#         if "dssp_condition" in data:
#             dssp = data["dssp_condition"]
#             dssp_mask = dssp != 3
#         dssp_mean = jnp.zeros((aa.shape[0], 3), dtype=jnp.float32)
#         dssp_mean_mask = jnp.zeros(aa.shape, dtype=jnp.bool_)
#         if "dssp_mean" in data:
#             dssp_mean = jnp.repeat(data["dssp_mean"], index_count(data["chain_index"], data["mask"]), axis=0)
#         # TODO
#         # condition = cond(aa, dssp, jnp.zeros((aa.shape[0], 14, 3), dtype=jnp.float32), jnp.zeros((aa.shape[0], 14), dtype=jnp.bool_),
#         #                  data["residue_index"], data["chain_index"], data["batch_index"], do_condition=dict(
#         #                      aa=aa, dssp=dssp, dssp_mean=dssp_mean, dssp_mean_mask=dssp_mean_mask
#         #                  ))

#         if c.pair_condition:
#             chain = data["chain_index"]
#             other_chain = chain[:, None] != chain[None, :]
#             pair_condition = jnp.concatenate((
#                 other_chain[..., None],
#                 jnp.zeros((chain.shape[0], chain.shape[0], 3)),
#             ), axis=-1)
#             # FIXME: make this input dependent
#             # relative_hotspot = jnp.zeros_like(other_chain)
#             # relative_hotspot = relative_hotspot.at[:, 15:20].set(1)
#             # relative_hotspot = jnp.where(other_chain, relative_hotspot, 0)
#             # pair_condition = pair_condition.at[:, :, 1].set(relative_hotspot)
#             # pair_condition = pair_condition.at[:, :, 2].set(relative_hotspot.T)
#             result["pair_condition"] = pair_condition
#         return result

#     def apply_diffusion(self, data):
#         c = self.config
#         chain = data["chain_index"]
#         batch = data["batch_index"]
#         mask = data["mask"]
#         pos = data["pos"] - index_mean(data["pos"][:, 1], batch, data["mask"][..., None])[:, None]
#         pos = Vec3Array.from_array(pos)
#         seq = data["seq"]
#         # drop_key = hk.next_rng_key() # FIXME
#         if c.diffusion_kind in ("edm", "fedm"):
#             sigma_pos = data["t_pos"]
#             pos_noised = diffuse_coordinates_edm(
#                 hk.next_rng_key(), pos, batch, sigma_pos, symm=c.symm)
#             sigma_seq = data["t_seq"]
#             seq_noised = diffuse_features_edm(
#                 hk.next_rng_key(), seq, sigma_seq)
#             t_pos = sigma_pos
#             t_seq = sigma_seq
#         if c.diffusion_kind == "flow":
#             t_pos = data["t_pos"]
#             t_seq = data["t_seq"]
#             pos_noised = diffuse_coordinates_blend(
#                 hk.next_rng_key(), pos, mask, batch, t_pos, scale=c.sigma_data)
#             seq_noised = seq
#         if c.diffusion_kind in ("vp", "vpfixed"):
#             # diffuse_coordinates = diffuse_atom_cloud
#             scale = None
#             t_pos = data["t_pos"]
#             t_seq = data["t_seq"]
#             sigma = SIGMA_SCALE[c.diffusion_time_scale](t_pos)
#             # s = 0.01
#             # alpha_bar = jnp.cos((t_pos + s) / (1 + s) * jnp.pi / 2) ** 2
#             # alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
#             # alpha_bar = jnp.clip(alpha_bar, 0, 1)
#             # sigma = jnp.sqrt(1 - alpha_bar)
#             t_pos = sigma
#             cloud_std = None
#             if "cloud_std" in data:
#                 cloud_std = data["cloud_std"][:, None, None]
#             complex_std = cloud_std
#             if "complex_std" in data:
#                 complex_std = data["complex_std"][:, None, None]
#             if c.diffusion_kind == "vpfixed":
#                 cloud_std = c.sigma_data
#             if c.correlated_cloud:
#                 pos_noised = diffuse_atom_chain_cloud(
#                     hk.next_rng_key(), pos, mask, chain, batch, t_pos,
#                     cloud_std=cloud_std, complex_std=complex_std, symm=c.symm)
#             else:
#                 pos_noised = diffuse_atom_cloud(
#                     hk.next_rng_key(), pos, mask, batch, t_pos,
#                     cloud_std=cloud_std, symm=c.symm)
#             seq_noised = diffuse_features_vp(
#                 hk.next_rng_key(), seq, t_seq,
#                 scale=c.sequence_noise_scale)
#         if c.sde:
#             pos_noised = Vec3Array.from_array(data["pos_noised"])
#         return dict(
#             seq_noised=seq_noised, pos_noised=pos_noised.to_array(),
#             t_pos=t_pos, t_seq=t_seq)

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
    else:
        result = {"in": 1.0, "out": sigma_data, "skip": 1.0,
                  "refine": 1, "loss": 1}
    if not c.refine:
        result["refine"] = 1.0
    return result

def make_rel(pos):
    pos = Vec3Array.from_array(pos)
    frames, _ = extract_aa_frames(pos)
    rel = frames[:, None, None].apply_inverse_to_point(pos[None, :, :])
    return rel.to_array()

def rel_features(rel, neighbours, d_max=22.0):
    index = jnp.arange(rel.shape[0], dtype=jnp.int32)
    rel = Vec3Array.from_array(rel[index[:, None], neighbours])
    distances = distance_rbf(rel.norm(), 0.0, d_max, 16)
    distances = distances.reshape(*distances.shape[:2], -1)
    directions = rel.normalized()
    directions = directions.reshape(*directions.shape[:2], -1)
    return jnp.concatenate((distances, directions), axis=-1)

def distogram_features(distogram, neighbours):
    index = jnp.arange(distogram.shape[0], dtype=jnp.int32)
    return jax.nn.softmax(distogram[index[:, None], neighbours], axis=-1)

class Distogram(hk.Module):
    def __init__(self, size, name: Optional[str] = "distogram"):
        super().__init__(name)
        self.size = size

    def __call__(self, local, resi, chain, batch, mask, cyclic_mask=None):
        left = Linear(32, bias=False, initializer=init_linear())(local)
        right = Linear(32, bias=False, initializer=init_linear())(local)
        pair = left[:, None] + right[None, :]
        pair += Linear(32, bias=False, initializer=init_linear())(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, None, cyclic_mask=cyclic_mask))
        pair = hk.LayerNorm([-1], True, True)(pair)
        result = MLP(32, self.size, depth=2,
                     activation=jax.nn.gelu,
                     bias=False, final_init=init_zeros())(pair)
        same_batch = batch[:, None] == batch[None, :]
        pair_mask = mask[:, None] * mask[None, :] * same_batch
        result = jnp.where(pair_mask[..., None], result, 0)
        return result

class Diffusion(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config
        self.diffusion_blocks = {
            "vp": VPDiffusionBlock
        }

    def __call__(self, data):
        c = self.config
        # enable selective cyclisation of chains
        cyclic_mask = None
        if c.cyclic and "cyclic_mask" in data:
            cyclic_mask = data["cyclic_mask"]
        diffusion_block = self.diffusion_blocks[c.block_type]
        diffusion_stack = DiffusionStack(c, diffusion_block)
        aa_head = Linear(20, initializer=init_zeros())
        dssp_head = Linear(3, initializer=init_zeros())
        distogram_head = Distogram(64)
        
        def model_iteration(data, prev, input_override=None):
            features = self.prepare_features(
                data, prev, input_override=input_override)
            local, pos, resi, chain, batch, mask = features
            local, pos, trajectory = diffusion_stack(
                local, pos, prev,
                resi, chain, batch, mask,
                cyclic_mask=cyclic_mask
            )
            aa = aa_head(local)
            dssp = dssp_head(local)
            distogram = distogram_head(
                local, resi, chain, batch, mask,
                cyclic_mask=cyclic_mask)
            prev = dict(
                aa=aa,
                dssp=dssp,
                distogram=distogram,
                local=local,
                pos=pos
            )
            return trajectory, prev

        def recycling_iter(i, prev):
            new_data = data
            new_pos = diffuse_coordinates_blend(
                hk.next_rng_key(), Vec3Array.from_array(data["pos"]),
                data["mask"], data["batch_index"], data["t_pos"]).to_array()
            new_seq, _ = diffuse_sequence(
                hk.next_rng_key(), data["aa_gt"], data["t_seq"][:, 0])
            _, prev = model_iteration(new_data, prev, input_override=(new_pos, new_seq))
            return prev

        batch_size = data["aa_gt"].shape[0]
        init_prev = dict(
            aa=jnp.zeros((batch_size, 20), dtype=jnp.float32),
            dssp=jnp.zeros((batch_size, 3), dtype=jnp.float32),
            distogram=jax.nn.one_hot(63 * jnp.ones((batch_size, batch_size), dtype=jnp.float32), 64, axis=-1),
            local=jnp.zeros((batch_size, c.local_size), dtype=jnp.float32),
            pos=data["pos_noised"]
        )
        prev = init_prev
        count = jax.random.randint(hk.next_rng_key(), (), 0, 2)
        prev = jax.lax.stop_gradient(hk.fori_loop(0, count, recycling_iter, prev))
        refine_count = jax.random.bernoulli(hk.next_rng_key(), 0.1, ()).astype(jnp.int32)
        init = jax.lax.stop_gradient(hk.fori_loop(0, refine_count, recycling_iter, prev))
        data["pos_noised"] = jnp.where(refine_count > 0, init["pos"], data["pos_noised"])
        data["seq_noised"] = jnp.where(refine_count > 0,
                                       jax.random.categorical(hk.next_rng_key(), init["aa"], axis=-1),
                                       data["seq_noised"])
        trajectory, prev = model_iteration(data, prev)
        pos = prev["pos"]
        local = prev["local"]

        # predict & handle sequence, losses etc.
        result = dict()
        # trajectory
        if isinstance(trajectory, tuple):
            trajectory, feature_trajectory = trajectory
            result["feature_trajectory"] = feature_trajectory
        result["trajectory"] = trajectory
        result["pos"] = pos
        result["distogram"] = jax.nn.log_softmax(prev["distogram"], axis=-1)
        # sequence feature update
        result["aa"] = jax.nn.log_softmax(prev["aa"], axis=-1)
        result["dssp"] = jax.nn.log_softmax(prev["dssp"], axis=-1)

        # decoder features and logits
        result["corrupt_aa"] = data["corrupt_aa"] * data["mask"] * (data["aa_gt"] != 20)
        result["aa_gt"] = data["aa_gt"]
        result["pos_noised"] = data["pos_noised"]
        # generate all-atom positions using predicted
        # side-chain torsion angles:
        aatype = data["aa_gt"]
        raw_angles, angles, angle_pos = get_angle_positions(
            aatype, local, pos)
        result["raw_angles"] = raw_angles
        result["angles"] = angles
        result["atom_pos"] = angle_pos
        return result

    def prepare_features(self, data, prev, input_override=None):
        c = self.config
        pos = data["pos_noised"]
        seq = data["seq_noised"]
        if input_override is not None:
            pos, seq = input_override
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        frames, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        local = Linear(c.local_size, initializer=init_linear())(
            jax.nn.one_hot(seq, 21, axis=-1))
        local += Linear(c.local_size, initializer=init_linear())(
            local_pos.to_array().reshape(local_pos.shape[0], -1))
        local += Linear(c.local_size, initializer=init_linear())(
            jax.nn.softmax(prev["aa"], axis=-1))
        local += Linear(c.local_size, initializer=init_linear())(
            jax.nn.softmax(prev["dssp"], axis=-1))
        local += Linear(c.local_size, initializer=init_linear())(
            hk.LayerNorm([-1], True, True)(prev["local"]))
        local = hk.LayerNorm([-1], True, True)(local)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local)
        local = hk.LayerNorm([-1], True, True)(local)

        return local, pos, resi, chain, batch, mask

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
        late_mask = data["t_pos"][:, 0] < 0.5
        late_mask *= mask
        
        # AA NLL loss
        if c.aa_weight:
            aa_predict_mask = mask * (data["aa_gt"] != 20) * result["corrupt_aa"]
            aa_predict_weight = 1 / jnp.maximum(aa_predict_mask.sum(keepdims=True), 1)
            aa_predict_weight = jnp.where(aa_predict_mask, aa_predict_weight, 0)
            aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
            aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
            aa_nll = (aa_nll * aa_predict_weight).sum()
            losses["aa"] = aa_nll
            total += c.aa_weight * aa_nll

        # DSSP NLL loss
        dssp_nll = -(result["dssp"] * jax.nn.one_hot(data["dssp"], 3, axis=-1)).sum(axis=-1)
        dssp_nll = jnp.where(mask, dssp_nll, 0).sum() / jnp.maximum(mask.sum(), 1)
        losses["dssp"] = dssp_nll
        total += 1.0 * dssp_nll

        # distogram NLL loss
        pos_gt = data["pos"]
        same_batch = data["batch_index"][:, None] == data["batch_index"][None, :]
        pair_mask = same_batch * mask[:, None] * mask[None, :]
        distance_gt = jnp.linalg.norm(pos_gt[:, None, 1] - pos_gt[None, :, 1], axis=-1)#.min(axis=(-1, -2))
        distogram_gt = distance_one_hot(distance_gt, 0.0, 22.0, 64)
        distogram_nll = -(result["distogram"] * distogram_gt).sum(axis=-1)
        distogram_nll = jnp.where(pair_mask, distogram_nll, 0).sum() / jnp.maximum(pair_mask.sum(), 1)
        losses["distogram"] = distogram_nll
        total += 1.0 * distogram_nll

        # # local NLL
        # pos_gt = Vec3Array.from_array(pos_gt)
        # pos_pred = Vec3Array.from_array(result["pos"])
        # frames, _ = extract_aa_frames(pos_gt)
        # pred_frames, _ = extract_aa_frames(pos_pred)
        # index = jnp.arange(pair_mask.shape[0], dtype=jnp.int32)
        # neighbours_gt = jnp.argsort(jnp.where(pair_mask, distance_gt, jnp.inf), axis=1)[:, :32]
        # neighbours_gt = jnp.where(pair_mask[index[:, None], neighbours_gt] < 20.0, neighbours_gt, -1)
        # neighbour_mask = (neighbours_gt != -1) * mask[:, None] * mask[neighbours_gt]
        # local_gt = frames[:, None, None].apply_inverse_to_point(pos_gt[neighbours_gt])
        # local_pred = pred_frames[:, None, None].apply_inverse_to_point(pos_pred[neighbours_gt])
        # local_error = jnp.clip((local_gt - local_pred).norm(), 0, 10.0) / 10.0
        # local_error = jnp.where(neighbour_mask, local_error, 0).sum() / jnp.maximum(neighbour_mask.sum(), 1)
        # losses["local"] = local_error
        # total += 0.1 * local_error

        # diffusion losses
        base_weight = mask / jnp.maximum(index_sum(mask.astype(jnp.float32), batch, mask), 1) / (batch.max() + 1)
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
            dist2 = jnp.clip(dist2, 0, do_clip[:, None]).mean(axis=-1)
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

def extract_neighbours(num_neighbours=64):
    if not isinstance(num_neighbours, int):
        raise TypeError(
            f"extract_neighbours requires an argument of type 'int'." 
            f"Got {type(num_neighbours)}.")
    def inner(pos, prev, resi, chain, batch, mask):
        same_batch = batch[:, None] == batch[None, :]
        same_chain = chain[:, None] == chain[None, :]
        same_chain *= same_batch
        mask = mask[:, None] * mask[None, :] * same_batch
        # perfectly extended chain distances
        distance = jnp.where(same_chain, abs(resi[:, None] - resi[None, :]) * 3.81, jnp.inf)
        # minimum of extended distance and predicted distogram distance
        mean_distogram = distance_one_hot_inverse(
                    jax.nn.softmax(prev["distogram"]), 0.0, 22.0, 64)
        distance = jnp.minimum(distance, jnp.where(mean_distogram < 8.0, mean_distogram, jnp.inf))
        # minimum of perfect distance and actual distance
        distance = jnp.minimum(distance, (pos[:, None, 1] - pos[None, :, 1]).norm())
        # minimum of distance and previous distance
        ppos = Vec3Array.from_array(prev["pos"])
        distance = jnp.minimum(distance, (ppos[:, None, 1] - ppos[None, :, 1]).norm())
        # set infinite distance between different-batch pairs
        distance = jnp.where(same_batch, distance, jnp.inf)
        # get 1 / distance ** 3 weighting
        log_p = -3 * distance
        randomness = jax.random.uniform(hk.next_rng_key(), distance.shape)
        gumbel = -jnp.log(-jnp.log(randomness + 1e-6) + 1e-6)
        random_distance = -(log_p + gumbel)
        neighbours = get_neighbours(num_neighbours)(random_distance, mask, None)
        print("LOLWTF", neighbours.shape, num_neighbours, random_distance.shape)
        return neighbours
    return inner

def distance_one_hot_inverse(p, d_min, d_max, bins):
    step = (d_max - d_min) / bins
    centers = d_min + jnp.arange(bins) * step + step / 2
    return (centers * p).sum(axis=-1)

class DiffusionStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, prev,
                 resi, chain, batch, mask,
                 cyclic_mask=None):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                neighbours = extract_neighbours(64)(
                    Vec3Array.from_array(pos), prev,
                    resi, chain, batch, mask)
                # run the diffusion block
                features, pos = block(c)(
                    features, pos, prev, neighbours,
                    resi, chain, batch, mask,
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
            trajectory = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        return local, pos, trajectory

class VPDiffusionBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, prev, neighbours,
                 resi, chain, batch, mask, cyclic_mask=None):
        c = self.config
        residual_update, residual_input, local_shape = get_residual_gadgets(
            features, c.resi_dual)
        _, current_local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
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
        pair = Linear(c.pair_size, bias=False)(
            distance_features(Vec3Array.from_array(pos), neighbours, d_max=22.0))
        pair += Linear(c.pair_size, bias=False)(
            rotation_features(extract_aa_frames(Vec3Array.from_array(pos))[0], neighbours))
        pair += Linear(c.pair_size, bias=False)(
            direction_features(Vec3Array.from_array(pos), neighbours))
        pair += Linear(c.pair_size, bias=False)(
            distance_features(Vec3Array.from_array(prev["pos"]), neighbours, d_max=22.0))
        pair += Linear(c.pair_size, bias=False)(
            rotation_features(extract_aa_frames(Vec3Array.from_array(prev["pos"]))[0], neighbours))
        pair += Linear(c.pair_size, bias=False)(
            direction_features(Vec3Array.from_array(prev["pos"]), neighbours))
        pair += Linear(c.pair_size, bias=False)(
            distogram_features(prev["distogram"], neighbours))
        pair += Linear(c.pair_size, bias=False)(
            sequence_relative_position(32, one_hot=True, cyclic=c.cyclic)(
                resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
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
        split_scale_out = c.sigma_data
        pos = update_positions(pos, local_norm,
                               scale=split_scale_out,
                               symm=c.symm)
        return features, pos.astype(local.dtype)

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
