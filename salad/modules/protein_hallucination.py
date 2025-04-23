# This module explores the possibility of latent-space
# protein diffusion models by joint structure prediction
# and denoising from a learned embedding of protein structures.
# - we embed proteins as #amino acids × #features latent variables
# - we add noise to this embedding according to a diffusion
#   noise schedule (VP or EDM-VE)
# - we train a joint decoder-denoiser to reconstruct the original
#   protein structure (with FAPE loss) and de-noise the noisy latent
#   variables (with noise prediction loss)
# As an edge case relating these models to hallucination-based
# approaches to protein design, we alternatively provide a decoder
# which is trained to predict pLDDT and pAE.
# We demonstrate that such a decoder can generate designable protein
# backbones by gradient descent on its pLDDT and pAE outputs.

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
    index_mean, index_sum, extract_aa_frames,
    extract_neighbours, distance_rbf, assign_sse,
    unique_chain, positions_to_ncacocb, distance_one_hot,
    single_protein_sidechains, compute_pseudo_cb)

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
    sequence_relative_position,
    sum_equivariant_pair_embedding,
    distance_features, rotation_features
)

# diffusion processes
from salad.modules.utils.diffusion import (
    diffuse_features_permute, diffuse_sequence,
    fourier_time_embedding
)

# modules
from salad.modules.protein_structure_diffusion import assign_dssp

class Hallucination(hk.Module):
    def __init__(self, config,
                 name: Optional[str] = "hallucination"):
        super().__init__(name)
        self.config = config

    def prepare_data(self, data):
        c = self.config
        pos = data["all_atom_positions"]
        atom_mask = data["all_atom_mask"]
        residue_gt = data["aa_gt"]
        chain = data["chain_index"]
        batch = data["batch_index"]

        # positions come in atom24 format - transform into atom14
        pos = pos[:, :14]
        atom_mask = atom_mask[:, :14]

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
        pos = jnp.where(
            atom_mask[..., None], pos,
            compute_pseudo_cb(pos)[:, None, :])

        # if specified, augment backbone with encoded positions
        data["chain_index"] = chain
        data["pos"] = pos
        data["atom_mask"] = atom_mask
        data["mask"] = mask
        latent = self.encode(data)

        # set all-atom-position target
        atom_pos = pos
        atom_mask = atom_mask
        return dict(latent=latent, pos=pos, chain_index=chain, mask=mask,
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
        # compute normalized latent vectors for each amino acid
        latent = Encoder(c)(data)
        latent = hk.LayerNorm([-1], False, False)(latent)
        return latent

    def apply_diffusion(self, data):
        c = self.config
        batch = data["batch_index"]
        mask = data["mask"]
        latent = data["latent"]
        time = jax.random.uniform(
            hk.next_rng_key(), (batch.shape[0], 1))[batch]
        # with probability p_clean, do not noise this example
        clean = jax.random.bernoulli(
            hk.next_rng_key(), c.p_clean, (batch.shape[0], 1))[batch]
        time = jnp.where(clean, 0.0, time)
        latent_noised = diffuse_features_permute(
            hk.next_rng_key(), latent, time, mask,
            scale=1.0)
        return dict(
            latent_noised=latent_noised,
            time=time)

    def __call__(self, data):
        c = self.config
        decoder = Decoder(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))

        result = decoder(data)
        err_result = decoder(jax.lax.stop_gradient(data))
        total, losses = decoder.loss(data, result, err_result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

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
        local = MLP(c.local_size * 4, c.local_size, activation=jax.nn.gelu,
                    bias=False)(
            jnp.concatenate(local_features, axis=-1))
        
        pair, pair_mask = sum_equivariant_pair_embedding(c, use_local=True)(
            local, positions, neighbours, resi, chain, batch, mask)
        positions = positions.to_array()
        local += SparseStructureMessage(c)(
            local, positions, pair, pair_mask,
            neighbours, resi, chain, batch, mask)
        local = hk.LayerNorm([-1], True, True)(local)
        return local, positions, neighbours, resi, chain, batch, mask

    def __call__(self, data):
        c = self.config
        local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data)
        local = EncoderStack(c, c.encoder_depth)(
            local, pos, neighbours, resi, chain, batch, mask)
        local = MLP(local.shape[-1] * 4, c.latent_size,
                    activation=jax.nn.gelu, bias=False,
                    final_init=init_glorot())(local)
        return local

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
    else:
        result = {"in": 1.0, "out": sigma_data, "skip": 1.0,
                  "refine": 1, "loss": 1 / sigma_data}
    if not c.refine:
        result["refine"] = 1.0
    return result

# implement FAPE-loss decoder in the spirit of the AlphaFold structure module
# without explicit frame manipulation.
# Should predict structure + distograms + relative pair orientation
# as well as predicted error for hallucination-mode usage
class Decoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data, predict_aa=False):
        c = self.config
        decoder_stack = DecoderStack(c, DecoderBlock)

        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]

        def apply_model(data, prev):
            features = self.prepare_features(data, prev)
            local, pos, resi, chain, batch, mask = features
            local, pos, trajectory = decoder_stack(
                local, pos, resi, chain, batch, mask)
            return local, pos, trajectory
        def iteration(index, prev):
            local, pos, _ = apply_model(data, prev)
            return (local, pos)
        count = jax.random.randint(hk.next_rng_key(), (), 0, 4)
        if c.eval:
            count = 3
        
        # construct "black hole initialized" idealised AA atom positions
        initial_frames = Rigid3Array.identity(data["batch_index"].shape)
        initial_pos, _ = single_protein_sidechains(
            jnp.zeros_like(data["batch_index"]), initial_frames,
            jnp.zeros((data["batch_index"].shape[0], 7, 2), dtype=jnp.float32))
        initial_pos = initial_pos.to_array()[:, :5]

        prev = (
            jnp.zeros((data["batch_index"].shape[0], c.local_size), dtype=jnp.float32), # prev local
            initial_pos # prev pos
        )
        # run recycling without gradients during training
        stop_when_training = jax.lax.stop_gradient
        if c.eval:
            stop_when_training = lambda x: x
        prev = stop_when_training(hk.fori_loop(0, count, iteration, prev))

        # final iteration with gradients
        local, pos, trajectory = apply_model(data, prev)

        # predict & handle sequence, losses etc.
        result = dict()
        # trajectory
        result["trajectory"] = trajectory
        result["pos"] = pos

        # pair features
        outer = MLP(local.shape[-1] * 2, 16, activation=jax.nn.gelu, bias=False, final_init=init_linear())(local)
        outer = jnp.einsum("ic,jd->ijcd", outer, outer).reshape(outer.shape[0], outer.shape[0], -1)
        pair = Linear(
            32, bias=False,
            initializer=init_linear())(local)[:, None]
        pair += Linear(
            32, bias=False,
            initializer=init_linear())(local)[None, :]
        pair += Linear(32, bias=False)(outer)
        pair += Linear(
            32, bias=False,
            initializer=init_linear())(
                sequence_relative_position(count=32, one_hot=True)(
                    resi, chain, batch))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(
            pair.shape[-1] * 2, pair.shape[-1],
            bias=False, activation=jax.nn.gelu,
            final_init=init_linear())(pair)
        # distogram predictions
        distogram = MLP(32, 64, bias=False,
                        activation=jax.nn.gelu,
                        final_init=init_zeros())(pair)
        result["distogram"] = distogram
        # error predictions
        error = MLP(32, 64, bias=False,
                    activation=jax.nn.gelu,
                    final_init=init_zeros())(pair)
        result["error"] = error

        # decoder features and logits
        aa_logits = Linear(20, bias=False, initializer=init_zeros())(local)
        result["aa"] = jax.nn.log_softmax(aa_logits, axis=-1)
        result["corrupt_aa"] = data["mask"] * (data["aa_gt"] != 20)
        result["aa_gt"] = data["aa_gt"]
        # lddt, lsr
        result["lddt"] = jax.nn.log_softmax(MLP(
            local.shape[-1] * 2, 64, bias=False,
            activation=jax.nn.gelu, final_init=init_zeros())(local), axis=-1)
        result["lsr"] = jax.nn.log_softmax(MLP(
            local.shape[-1] * 2, 64, bias=False,
            activation=jax.nn.gelu, final_init=init_zeros())(local), axis=-1)
        # dssp
        result["dssp"] = jax.nn.log_softmax(Linear(3, initializer=init_zeros())(local), axis=-1)
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
        return result

    def prepare_features(self, data, prev):
        c = self.config
        prev_local, prev_pos = prev
        time = data["time"]
        latent = data["latent_noised"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        frames, local_pos = extract_aa_frames(
            Vec3Array.from_array(prev_pos))
        local_features = [
            latent,
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1)
        ]
        local_features = jnp.concatenate(
            local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
        local = hk.LayerNorm([-1], True, True)(local)
        local += MLP(local.shape[-1] * 2, local.shape[-1], activation=jax.nn.gelu,
                     bias=False, final_init=init_zeros())(hk.LayerNorm([-1], True, True)(prev_local))
        pos = prev_pos

        return local, pos, resi, chain, batch, mask

    def loss(self, data, result, err_result):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        losses = dict()
        total = 0.0
        # AA NLL loss
        aa_predict_mask = mask * (data["aa_gt"] != 20) * result["corrupt_aa"]
        aa_predict_weight = 1 / jnp.maximum(aa_predict_mask.sum(keepdims=True), 1e-6)
        aa_predict_weight = jnp.where(aa_predict_mask, aa_predict_weight, 0)
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = (aa_nll * aa_predict_weight).sum()
        losses["aa"] = aa_nll
        total += aa_nll

        # DSSP loss
        dssp_gt, _, _ = assign_dssp(data["all_atom_positions"], batch, mask)
        dssp_nll = -(result["dssp"] * jax.nn.one_hot(dssp_gt, 3, axis=-1)).sum(axis=-1)
        dssp_nll = jnp.where(mask, dssp_nll, 0).sum() / jnp.maximum(mask.sum(), 1)
        losses["dssp"] = dssp_nll
        total += dssp_nll

        # diffusion losses
        base_weight = mask / jnp.maximum(index_sum(mask.astype(jnp.float32), batch, mask), 1) / (batch.max() + 1)
        # random clipping of positional losses
        do_clip = jnp.where(
            jax.random.bernoulli(hk.next_rng_key(), 0.7, batch.shape)[batch],
            100.0,
            jnp.inf)
        # local / rotamer loss
        gt_frames, _ = extract_aa_frames(
            Vec3Array.from_array(data["atom_pos"]))
        frames, _ = extract_aa_frames(
            Vec3Array.from_array(result["atom_pos"]))
        frames, local_positions = jax.vmap(extract_aa_frames)(
            Vec3Array.from_array(result["atom_trajectory"]))
        gt_local_positions = gt_frames[:, None].apply_inverse_to_point(
            Vec3Array.from_array(data["pos"]))
        frames = frames[-1]
        gt_local_positions = gt_frames[:, None].apply_inverse_to_point(
            Vec3Array.from_array(data["atom_pos"]))
        local_positions = frames[:, None].apply_inverse_to_point(
            Vec3Array.from_array(result["atom_pos"]))
        dist2 = (local_positions - gt_local_positions).norm2()
        dist2 = jnp.clip(dist2, 0, do_clip[:, None])
        rotamer = (dist2 * data["atom_mask"]).sum(axis=-1) / 3
        rotamer /= jnp.maximum(data["atom_mask"].sum(axis=-1), 1)
        rotamer = (rotamer * base_weight).sum()
        losses["rotamer"] = rotamer
        total += 10.0 * rotamer

        # pair losses
        pair_mask = mask[:, None] * mask[None, :]
        pair_mask *= data["batch_index"][:, None] == data["batch_index"][None, :]
        # distogram loss
        cb_gt = data["atom_pos"][:, 4]
        dist_gt = jax.lax.stop_gradient(jnp.linalg.norm(cb_gt[:, None] - cb_gt[None, :], axis=-1))
        distogram_gt = distance_one_hot(dist_gt, 0.0, 22.0, bins=64)
        distogram_nll = -(jax.nn.log_softmax(result["distogram"], axis=-1) * distogram_gt).sum(axis=-1)
        distogram_nll = jnp.where(pair_mask, distogram_nll, 0).sum() / jnp.maximum(pair_mask.sum(), 1)
        losses["distogram"] = 1.0 * distogram_nll
        total += 1.0 * distogram_nll

        # FAPE
        trajectory_fape, _, _, _ = hal_atom_fape(
            Vec3Array.from_array(result["atom_trajectory"]),
            Vec3Array.from_array(data["all_atom_positions"]),
            data["aa_gt"],
            batch, data["chain_index"],
            data["atom_mask"]
        )
        fape, _, final_ae, _ = all_atom_fape(
            Vec3Array.from_array(result["atom_trajectory"][-1]),
            Vec3Array.from_array(data["all_atom_positions"]),
            data["aa_gt"],
            batch, data["chain_index"],
            data["atom_mask"]
        )
        trajectory_fape_loss = trajectory_fape.mean()
        losses["trajectory_fape"] = trajectory_fape_loss
        losses["fape"] = fape
        total += 10.0 * (2 * fape + trajectory_fape_loss)

        # error loss
        error_gt = distance_one_hot(final_ae, bins=64)
        error_loss = -(jax.nn.log_softmax(err_result["error"], axis=-1) * error_gt).sum(axis=-1)
        error_loss = jnp.where(pair_mask, error_loss, 0).sum() / jnp.maximum(pair_mask.sum(), 1)
        losses["error"] = error_loss
        total += 0.01 * error_loss

        # lddt loss
        gt_lddt = lddt(
            Vec3Array.from_array(result["trajectory"][-1]),
            Vec3Array.from_array(data["all_atom_positions"]), mask, cutoff=15.0)
        mean_gt_lddt = ((gt_lddt * mask).sum(axis=-1) / jnp.maximum(mask.sum(axis=-1), 1e-6)).mean()
        gt_lddt = distance_one_hot(gt_lddt, 0, 1, bins=64)
        lddt_nll = -(jax.nn.log_softmax(result["lddt"], axis=-1) * gt_lddt).sum(axis=-1)
        lddt_nll = jnp.where(mask, lddt_nll, 0)
        lddt_nll = lddt_nll.sum(axis=-1) / jnp.maximum(mask.sum(axis=-1), 1e-6)
        lddt_nll = lddt_nll.mean()
        losses["mean_lddt"] = mean_gt_lddt
        total += 0.01 * lddt_nll
        
        # lsr loss
        gt_lsr = lsr(5 * result["aa"][-1], data["aa_gt"],
                     Vec3Array.from_array(data["all_atom_positions"]), mask, cutoff=15.0)
        mean_gt_lsr = ((gt_lsr * mask).sum(axis=-1) / jnp.maximum(mask.sum(axis=-1), 1e-6)).mean()
        gt_lsr = jax.lax.stop_gradient(distance_one_hot(gt_lsr, 0, 1, bins=64))
        lsr_nll = -(jax.nn.log_softmax(result["lsr"], axis=-1) * gt_lsr).sum(axis=-1)
        lsr_nll = jnp.where(mask, lsr_nll, 0)
        lsr_nll = lsr_nll.sum(axis=-1) / jnp.maximum(mask.sum(axis=-1), 1e-6)
        lsr_nll = lsr_nll.mean()
        losses["mean_lsr"] = mean_gt_lsr
        total += 0.01 * lsr_nll

        # violation loss
        res_mask = data["mask"]
        pred_mask = get_atom14_mask(data["aa_gt"]) * res_mask[:, None]
        # FIXME: ensure our data is in atom14 format
        violation, _ = violation_loss(data["aa_gt"],
                                      data["residue_index"],
                                      result["atom_pos"],
                                      pred_mask,
                                      res_mask,
                                      clash_overlap_tolerance=1.5,
                                      violation_tolerance_factor=12.0,
                                      chain_index=data["chain_index"],
                                      batch_index=data["batch_index"],
                                    per_residue=False)
        losses["violation"] = violation.mean()
        total += 0.01 * violation.mean()
        return total, losses

def lddt(positions, gt_positions, mask, cutoff=15.0):
    gt_positions = gt_positions[..., 1]
    positions = positions[..., 1]
    gt_distance = (gt_positions[:, None] - gt_positions[None, :]).norm()
    distance_mask = (gt_distance < cutoff) * mask[:, None] * mask[None, :] * (1 - jnp.eye(gt_distance.shape[1]))
    distance_mask = distance_mask.astype(jnp.bool_)
    distance = (positions[:, None] - positions[None, :]).norm()
    dist_error = jnp.where(distance_mask, jnp.abs(gt_distance - distance), jnp.inf)
    lddt = (dist_error[..., None] < jnp.array([0.5, 1.0, 2.0, 4.0])).astype(jnp.float32).mean(axis=-1)
    lddt = jnp.where(distance_mask, lddt, 0).sum(axis=-1)
    lddt /= jnp.maximum(distance_mask.astype(jnp.float32).sum(axis=-1), 1e-6)
    return lddt

def lsr(aa, aa_gt, gt_positions, mask, cutoff=15.0):
    gt_positions = gt_positions[..., 1]
    gt_distance = (gt_positions[:, None] - gt_positions[None, :]).norm()
    distance_mask = (gt_distance < cutoff) * mask[:, None] * mask[None, :] * (1 - jnp.eye(gt_distance.shape[1]))
    distance_mask = distance_mask.astype(jnp.bool_)
    recovery_percentage = (jax.nn.softmax(aa, axis=-1) * jax.nn.one_hot(aa_gt, 20, axis=-1)).sum(axis=-1)
    local_recovery = (recovery_percentage[None, :] * distance_mask).sum(axis=-1)
    local_recovery /= jnp.maximum(distance_mask.astype(jnp.float32).sum(axis=-1), 1e-6)
    return local_recovery

def hal_atom_fape(prediction: Vec3Array, ground_truth: Vec3Array, aa_type: jnp.ndarray,
                  batch: jnp.ndarray, chain: jnp.ndarray, all_atom_mask: jnp.ndarray) -> jnp.ndarray:
    def get_renamings(aa_type, ground_truth, all_atom_mask, prediction):
        if ground_truth.shape[-1] != 14:
            gt_positions, gt_mask = atom37_to_atom14(aa_type, ground_truth, all_atom_mask)
        else:
            gt_positions = ground_truth
            gt_mask = all_atom_mask
        alt_gt_positions, alt_gt_mask = get_alt_atom14(aa_type, gt_positions, gt_mask)
        atom_is_ambiguous = get_atom14_is_ambiguous(aa_type)
        alt_naming_is_better = find_optimal_renaming(
            gt_positions=gt_positions,
            alt_gt_positions=alt_gt_positions,
            atom_is_ambiguous=atom_is_ambiguous,
            gt_exists=gt_mask,
            pred_positions=prediction)
        use_alt = alt_naming_is_better[:, None]

        all_atom_mask = (1. - use_alt) * gt_mask + use_alt * alt_gt_mask
        gt_positions = (1. - use_alt) * gt_positions + use_alt * alt_gt_positions
        return gt_positions, all_atom_mask
    ground_truth, all_atom_mask = get_renamings(aa_type, ground_truth, all_atom_mask, prediction[-1])
    same_item = batch[:, None] == batch[None, :]
    pair_mask = all_atom_mask[:, None] * all_atom_mask[None, :] * same_item[..., None]
    prediction_shape = prediction.shape
    prediction = prediction.to_array().reshape(-1, *prediction.shape[2:], 3)
    prediction = Vec3Array.from_array(prediction)
    pframes, _ = extract_aa_frames(prediction)
    pframes = pframes.to_array()
    pframes = pframes.reshape(*prediction_shape[:-1], *pframes.shape[-2:])
    pframes = Rigid3Array.from_array(pframes)
    prediction = prediction.to_array().reshape(*prediction_shape, 3)
    prediction = Vec3Array.from_array(prediction)
    gframes, _ = extract_aa_frames(ground_truth)
    pglobal = pframes[:, :, None, None].apply_inverse_to_point(prediction[:, None, :, :])
    gglobal = gframes[:, None, None].apply_inverse_to_point(ground_truth[None, :, :])
    fape = jnp.where(pair_mask[None, ...], (pglobal - gglobal[None]).norm(), 0)
    fape = fape.sum(axis=-1) / jnp.maximum(pair_mask[None, ...].sum(axis=-1), 1e-6)
    rotamer_error = jnp.diagonal(fape, 0, -2, -1)
    rotamer_loss = rotamer_error.sum(axis=-1) / jnp.maximum(all_atom_mask.any(axis=-1).sum(axis=-1), 1e-6)
    rotamer_error = rotamer_error[-1]
    pair_mask = pair_mask.any(axis=-1)
    final_ae = fape[-1]
    same_chain = (chain[:, None] == chain[None, :]) * pair_mask
    other_chain = (chain[:, None] != chain[None, :]) * pair_mask
    intra_coin = jnp.zeros_like(jax.random.bernoulli(hk.next_rng_key(), 0.1, batch.shape))
    inter_coin = jnp.zeros_like(jax.random.bernoulli(hk.next_rng_key(), 0.1, batch.shape))
    intra_fape = jnp.where(same_chain[None], jnp.clip(fape, 0.0, jnp.where(intra_coin, jnp.inf, 10.0)), 0).sum(axis=(-1, -2))
    intra_fape /= jnp.maximum((pair_mask * same_chain)[None].sum(axis=(-1, -2)), 1e-6)
    intra_fape /= 10.0
    inter_fape = jnp.where(other_chain[None], jnp.clip(fape, 0.0, jnp.where(inter_coin, jnp.inf, 10.0)), 0).sum(axis=(-1, -2))
    inter_fape /= jnp.maximum((pair_mask * other_chain)[None].sum(axis=(-1, -2)), 1e-6)
    inter_fape /= 10.0
    fape = intra_fape + inter_fape
    return fape, rotamer_loss, final_ae, rotamer_error

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

class DecoderStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos,
                 resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # extract neighbours once for each block
                if c.dense_decoder:
                    neighbours = None
                else:
                    neighbours = extract_neighbours(
                        num_index=c.index_neighbours,
                        num_spatial=c.spatial_neighbours,
                        num_random=c.random_neighbours)(
                            Vec3Array.from_array(pos),
                            resi, chain, batch, mask)
                # run the diffusion block
                features, pos = block(c)(
                    features, pos,
                    neighbours,
                    resi, chain, batch, mask)
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), pos
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
            trajectory = trajectory.reshape(-1, *trajectory.shape[2:])
        return local, pos, trajectory

class DecoderBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos,
                 neighbours,
                 resi, chain, batch, mask):
        c = self.config
        residual_update, residual_input, local_shape = get_residual_gadgets(
            features, c.resi_dual)
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        # add local position features to local features
        features = residual_update(
            features,
            PositionToLocal(c.local_size)(
                local_pos.to_array()))
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
            sequence_relative_position(32, one_hot=True)(resi, chain, batch, neighbours))
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
                               scale=split_scale_out)
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
    if neighbours is None:
        return mask[:, None] * mask[None, :]
    return mask[:, None] * mask[neighbours] * (neighbours != -1)

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

def update_positions(pos, local_norm, scale=10.0):
    frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
    pos_update = scale * Linear(
        pos.shape[-2] * 3, initializer=init_zeros(),
        bias=False)(local_norm)
    pos_update = Vec3Array.from_array(
        pos_update.reshape(*pos_update.shape[:-1], -1, 3))
    local_pos += pos_update

    # project updated pos to global coordinates
    pos = frames[..., None].apply_to_point(local_pos).to_array()
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
