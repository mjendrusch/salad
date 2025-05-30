"""This module implements the protein structure denoising models from the manuscript."""

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
    extract_neighbours, distance_rbf, distance_one_hot,
    unique_chain, positions_to_ncacocb, axis_index,
    single_protein_sidechains, compute_pseudo_cb,
    get_random_neighbours, get_spatial_neighbours,
    get_index_neighbours, get_neighbours, index_align)

from salad.modules.utils.dssp import assign_dssp, drop_dssp

# import data augmentation utilities & layerwise distogram utilities
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
    DenseNonEquivariantPointAttention,
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
    r"""Computes noise standard deviation scale for VP diffusion with a cosine schedule.

    Args:
        time: raw diffusion time between 0.0 (no noise) 1nd 1.0 (100% noise).

    Returns:
        Noise standard deviation scale for VP diffusion.
    """
    alpha_bar = jnp.cos((time + s) / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar /= jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar = jnp.clip(alpha_bar, 0, 1)
    sigma = jnp.sqrt(1 - alpha_bar)
    return sigma


def sigma_scale_framediff(time, bmin=0.1, bmax=20.0):
    r"""Computes noise standard deviation scale for the noise schedule used by FrameDiff."""
    Gs = time * bmin + 0.5 * time**2 * (bmax - bmin)
    sigma = jnp.sqrt(1 - jnp.exp(-Gs))
    return sigma


def sigma_scale_none(time):
    r"""Computes noise standard deviation schale for a linear noise schedule."""
    return time


# constant dictionary of standard deviation scales.
SIGMA_SCALE = dict(
    cosine=sigma_scale_cosine,
    framediff=sigma_scale_framediff,
    none=sigma_scale_none,
    linear=sigma_scale_none
)


class StructureDiffusion(hk.Module):
    r"""Wrapper class for SALAD structure diffusion models."""
    def __init__(self, config, name: Optional[str] = "structure_diffusion"):
        r"""Construct a SALAD model wrapper for training."""
        super().__init__(name)
        self.config = config

    def prepare_data(self, data):
        r"""Preprocess a batch of input data for model training."""
        c = self.config
        # get position and chain information
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
        pos = jnp.where(atom_mask[..., None], pos, compute_pseudo_cb(pos)[:, None, :])
        # if running a semi-equivariant model (some non-equivariant features)
        # add rotational and translational data augmentation
        if c.equivariance == "semi_equivariant":
            pos = (
                structure_augmentation(
                    pos / c.sigma_data, data["batch_index"], data["mask"]
                )
                * c.sigma_data
            )
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
        return dict(
            seq=seq,  # [N, 20]int32 sequence information
            pos=pos,  # [N, 5 + augment_size, 3]float32, residue positions with 'augment_size' pseudoatoms
            chain_index=chain,  # [N]int32 chain index
            mask=mask,  # [N]bool residue mask
            atom_pos=atom_pos,  # [N, 14, 3]float32 per-residue atom14 format atom positions
            atom_mask=atom_mask,  # [N, 14]bool per-residue atom14 format atom mask
            all_atom_positions=pos_14,  # same as above
            all_atom_mask=atom_mask_14,  # same as above
        )

    def encode(self, data):
        r"""Encode position and amino acid information to produce sequence and
        position features for diffusion."""
        # get position and amino acid information
        pos = data["pos"]
        pos_14 = pos
        aatype = data["aa_gt"]
        c = self.config
        # set up N, CA, C, O + pseudo CB positions
        backbone = positions_to_ncacocb(pos)
        pos = backbone
        # for models with non-equivariant features
        # randomly rotate and translate the input data
        # for encoding
        if c.equivariance == "semi_equivariant":
            center, translation, rotation = structure_augmentation_params(
                pos / c.sigma_data, data["batch_index"], data["mask"]
            )
            pos = c.sigma_data * apply_structure_augmentation(
                pos / c.sigma_data, center, translation, rotation
            )
        # set the position field in the data dictionary to the
        # transformed N,CA,C,O,CB positions
        data["pos"] = pos
        if c.augment_size > 0:
            # add additional pseudo-atom positions to each residue
            _, augmented_pos = Encoder(c)(data)
            pos = augmented_pos
        elif c.encode_atom14:
            # directly featurize each residue by its 14 possible
            # atom positions, setting all non-existent atoms to
            # either a fixed position (the position of the CA atom)
            # or a learned position (using Atom14Encoder)
            pos = pos_14
            data["pos"] = pos
            if c.atom14_ca:
                pos = jnp.where(data["atom_mask"][..., None], pos, pos[:, 1:2])
            if c.atom14_learned:
                pos = Atom14Encoder(c)(data)
            # additional post-encoder data augmentation for non-equivariant models
            if c.equivariance == "semi_equivariant":
                center, translation, rotation = structure_augmentation_params(
                    pos / c.sigma_data, data["batch_index"], data["mask"]
                )
                pos = c.sigma_data * apply_structure_augmentation(
                    pos / c.sigma_data, center, translation, rotation
                )
        else:
            # otherwise, return N,CA,C,O,CB positions
            pos = backbone
        # invert the structure augmentation after the encoder
        # to return atom coordinates in the same orientation
        # as the input positions in `data`.
        if c.equivariance == "semi_equivariant":
            pos = c.sigma_data * apply_inverse_structure_augmentation(
                pos / c.sigma_data, center, translation, rotation
            )
        # set sequence features to the ground-truth amino acid sequence
        seq = aatype
        return seq, pos

    def prepare_condition(self, data):
        r"""Prepares conditioning information for training."""
        c = self.config
        aa = data["aa_gt"]
        pos = data["atom_pos"]
        pos_mask = data["atom_mask"]
        # assign 3-state secondary structure for the input 3D structure
        # returning the secondary structure (dssp), a block-index (blocks)
        # which groups contiguous secondary structure elements and
        # a block adjacency matrix (block_adjacency) which specifies
        # contacts between secondary structure elements.
        dssp, blocks, block_adjacency = assign_dssp(
            pos, data["batch_index"], pos_mask.any(axis=-1)
        )
        # randomly drop block contacts with probability p_drop_dssp
        _, block_adjacency = drop_dssp(
            hk.next_rng_key(), dssp, blocks, block_adjacency, p_drop=c.p_drop_dssp
        )
        # compute conditioning information and masks
        condition, aa_mask, dssp_mask = Condition(c)(
            aa, dssp, pos_mask.any(axis=-1),
            data["residue_index"], data["chain_index"],
            data["batch_index"], set_condition=None)

        # include conditioning stochastically with p = 0.5 during training
        use_condition = jax.random.bernoulli(hk.next_rng_key(), 0.5, (aa.shape[0],))
        use_condition = use_condition[data["batch_index"]]
        use_pair_condition = use_condition[:, None] * use_condition[None, :]

        condition = jnp.where(use_condition[..., None], condition, 0)
        result = dict(
            condition=condition, # condition embedding
            dssp_gt=dssp, # ground truth secondary structure
            aa_mask=jnp.where(use_condition, aa_mask, 0), # mask for amino acid identity conditioning
            dssp_mask=jnp.where(use_condition, dssp_mask, 0), # mask for secondary structure conditioning
        )

        # set up pair condition
        batch = data["batch_index"]
        chain = data["chain_index"]
        same_batch = batch[:, None] == batch[None, :]
        # set up CB atom distance map
        cb = Vec3Array.from_array(data["pos"][:, 4])
        dmap = (cb[:, None] - cb[None, :]).norm()
        # set up relative orientation map
        frames, _ = extract_aa_frames(Vec3Array.from_array(data["pos"]))
        rot = frames.rotation
        rot_inverse = frames.rotation.inverse()
        omap = (rot_inverse[:, None] @ rot[None, :]).to_array()
        omap = omap.reshape(cb.shape[0], cb.shape[0], -1)
        # randomly mask between 20 and 80 percent of structure conditioning information
        percentage = jax.random.uniform(
            hk.next_rng_key(), (cb.shape[0],), minval=0.2, maxval=0.8
        )[batch]
        struc_mask = jax.random.bernoulli(hk.next_rng_key(), percentage)
        struc_mask *= jax.random.bernoulli(hk.next_rng_key(), p=0.5, shape=batch.shape)[
            chain
        ]
        dmap_mask = (
            struc_mask[:, None] * struc_mask[None, :] * same_batch * use_pair_condition
        )
        # for multi-motif scaffolding:
        # segment the entire batch and accept segments for conditioning with p=0.5
        if c.multi_motif:
            total_size = batch.shape[0]
            index = jnp.arange(total_size)

            # iterates over an array of residue indices
            # and splits it into segments of a random length
            def body(carry):
                data, seg_pos, segment = carry
                size = jax.lax.dynamic_index_in_dim(
                    segment_size, seg_pos, keepdims=False
                )
                update_mask = (seg_pos <= index) * (index < seg_pos + size)
                data = jnp.where(update_mask, segment, data)
                seg_pos = seg_pos + size
                return data, seg_pos, segment + 1

            # condition for stopping the iteration over residue indices
            def cond(carry):
                _, seg_pos, _ = carry
                return seg_pos < total_size

            # initialize random segment sizes between 10 and 50 amino acids
            segment_size = jax.random.randint(hk.next_rng_key(), batch.shape, 10, 50)
            # initialize iteration state
            seg_pos = 0
            segment = 0
            segi = jnp.zeros_like(batch)
            init = (segi, seg_pos, segment)
            # iterate over residues and return the segment index (segi)
            segi, seg_pos, segment = jax.lax.while_loop(cond, body, init)
            # assign each segment into one of 2 segment groups (0 or 1)
            seg_group = jax.random.randint(hk.next_rng_key(), batch.shape, 0, 2)[segi]
            # drop segments from being used for conditioning with p = 0.5
            seg_active = jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)[segi]
            struc_mask = seg_active * pos_mask.any(axis=1)
            # make a distance map mask based on all active segments
            # in each group. No between-group distances are passed
            # to the model during training, simulating a multi-motif
            # scaffolding task.
            dmap_mask = (
                struc_mask[:, None]
                * struc_mask[None, :]
                * (seg_group[:, None] == seg_group[None, :])
            )
            dmap_mask *= same_batch

        # initialize pair conditioning information
        pair_flags = get_pair_condition(
            pos, aa, block_adjacency, pos_mask,
            data["residue_index"], data["chain_index"],
            data["batch_index"], set_condition=None)
        pair_flags = jnp.where(use_pair_condition[..., None], pair_flags, 0)
        result["pair_condition"] = dict(
            dmap=dmap, omap=omap, dmap_mask=dmap_mask, flags=pair_flags
        )

        return result

    def apply_diffusion(self, data):
        r"""Applies noise to input positions according to the diffusion process
        specified in the model configuration."""
        c = self.config
        # get position and sequence information
        batch = data["batch_index"]
        mask = data["mask"]
        pos = Vec3Array.from_array(data["pos"])
        seq = data["seq"]
        # when using variance-expanding diffusion
        if c.diffusion_kind == "edm":
            # sample a noise level for each structure in the batch
            # according to a log-normal distribution
            sigma_pos, _ = get_sigma_edm(
                batch,
                meanval=c.pos_mean_sigma,
                stdval=c.pos_std_sigma,
                minval=c.pos_min_sigma,
                maxval=c.pos_max_sigma,
            )
            # sample a noise level for each sequence in the batch
            # from a uniform distribution
            sigma_seq = jax.random.uniform(hk.next_rng_key(), (batch.shape[0],))[batch]
            if "t_pos" in data:
                t_pos = data["t_pos"]
                t_seq = data["t_seq"]
            # apply noise to the input structure
            pos_noised = diffuse_coordinates_edm(
                hk.next_rng_key(), pos, batch, sigma_pos[:, None]
            )
            # and sequence, if using sequence information
            seq_noised, corrupt_aa = diffuse_sequence(hk.next_rng_key(), seq, sigma_seq)
            t_pos = sigma_pos
            t_seq = sigma_seq
        # when using any type of diffusion process interpolating
        # between pure noise and a structure (VP and flow-matching)
        if c.diffusion_kind in ("vp", "vpfixed", "flow", "flow_scaled"):
            # sample a structure diffusion time from a uniform distribution
            t_pos = jax.random.uniform(hk.next_rng_key(), (batch.shape[0],))[batch]
            # compute the noise scale for that diffusion time
            sigma = SIGMA_SCALE[c.diffusion_time_scale](t_pos)
            t_pos = sigma
            # sample a sequence diffusion time from a uniform distribution
            t_seq = jax.random.uniform(hk.next_rng_key(), (batch.shape[0],))[batch]
            if "t_pos" in data:
                t_pos = data["t_pos"]
                t_seq = data["t_seq"]
            # set the standard deviation of the noise
            cloud_std = None
            # for fixed-variance noise this is fixed in the configuration
            if c.diffusion_kind in ("vpfixed", "flow"):
                cloud_std = c.sigma_data
            # apply noise to the input structure. If no cloud_std is
            # provided, use the standard devation of CA positions in
            # the input structure.
            pos_noised = diffuse_atom_cloud(
                hk.next_rng_key(), pos, mask, batch, t_pos[:, None],
                cloud_std=cloud_std, flow=c.diffusion_kind in ("flow", "flow_scaled"))
            # apply noise to the sequence
            seq_noised, corrupt_aa = diffuse_sequence(hk.next_rng_key(), seq, t_seq)
        return dict(
            seq_noised=seq_noised,
            pos_noised=pos_noised.to_array(),
            t_pos=t_pos,
            t_seq=t_seq,
            corrupt_aa=corrupt_aa,
        )

    def __call__(self, data):
        c = self.config
        diffusion = Diffusion(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # prepare condition / task information
        data.update(self.prepare_condition(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))

        # self-conditioning function
        def iteration_body(i, prev):
            override = self.apply_diffusion(data)
            result = diffusion(data, prev, override=override)
            prev = dict(pos=result["pos"], local=result["local"])
            return jax.lax.stop_gradient(prev)

        # randomly run the model with or without self-conditioning
        prev = diffusion.init_prev(data)
        if not hk.running_init():
            count = jax.random.randint(hk.next_rng_key(), (), 0, 2)
            prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = diffusion(data, prev)
        # compute losses
        total, losses = diffusion.loss(data, result)
        out_dict = dict(results=result, losses=losses)
        return total, out_dict


class StructureDiffusionInference(StructureDiffusion):
    """Inference wrapper for salad diffusion models."""
    def __init__(self, config, name: Optional[str] = "structure_diffusion"):
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

        # single model iteration
        def apply_model(prev):
            result = diffusion(data, prev, predict_aa=True)
            prev = dict(pos=result["pos"], local=result["local"])
            return result, prev

        # if no self-conditioning info was provided
        # initialize self-conditioning info to 0.
        if prev is None:
            prev = diffusion.init_prev(data)
        # run model
        result, prev = apply_model(prev)

        # return noisy input for debugging purposes
        result["pos_input"] = data["pos_noised"]
        if c.aa_trajectory:
            result["aa"] = result["aa_trajectory"][-1]
            result["aatype"] = jnp.argmax(result["aa_trajectory"][-1], axis=-1)

        return result, prev

    def prepare_condition(self, data):
        """Prepare condition information."""
        c = self.config
        result = dict()
        cond = Condition(c)
        # initialize amino acid conditioning
        aa = 20 * jnp.ones_like(data["aa_gt"])
        if "aa_condition" in data:
            aa = data["aa_condition"]
        # initialize secondary structure conditioning
        dssp = 3 * jnp.ones_like(data["aa_gt"])
        if "dssp_condition" in data:
            dssp = data["dssp_condition"]
        dssp_mean = jnp.zeros((aa.shape[0], 3), dtype=jnp.float32)
        dssp_mean_mask = jnp.zeros(aa.shape, dtype=jnp.bool_)
        if "dssp_mean" in data:
            dssp_mean = jnp.stack([data["dssp_mean"]] * aa.shape[0], axis=0)
            dssp_mean_mask = jnp.ones(aa.shape, dtype=jnp.bool_)
        # process everything into per-residue condition features
        condition, _, _ = cond(
            aa,
            dssp,
            data["mask"],
            data["residue_index"],
            data["chain_index"],
            data["batch_index"],
            set_condition=dict(
                aa=aa, dssp=dssp, dssp_mean=dssp_mean, dssp_mean_mask=dssp_mean_mask))
        result["condition"] = condition

        # initialize distance and orientation map conditioning information
        dmap = jnp.zeros((aa.shape[0], aa.shape[0]), dtype=jnp.float32)
        omap = jnp.zeros((aa.shape[0], aa.shape[0], 9), dtype=jnp.float32)
        dmap_mask = jnp.zeros_like(dmap, dtype=jnp.bool_)
        if "dmap_mask" in data:
            dmap = data["dmap"]
            dmap_mask = data["dmap_mask"]
            if "omap" in data:
                omap = data["omap"]

        # initialize residue pair flags
        # FIXME: currently we do not support block-contact conditioning
        # or hotspot conditioning.
        chain = data["chain_index"]
        chain_contacts = chain[:, None] != chain[None, :]
        if "chain_contacts" in data:
            chain_contacts = data["chain_contacts"]
        flags = jnp.concatenate(
            (
                chain_contacts[..., None],
                jnp.zeros((chain.shape[0], chain.shape[0], 2)),
            ),
            axis=-1,
        )
        pair_condition = dict(
            dmap=dmap,
            omap=omap,
            dmap_mask=dmap_mask,
            flags=flags,
        )
        result["pair_condition"] = pair_condition
        return result

    def apply_diffusion(self, data):
        """Apply noise to the input."""
        c = self.config
        batch = data["batch_index"]
        mask = data["mask"]
        pos = data["pos"]

        pos = Vec3Array.from_array(pos)
        seq = data["seq"]
        t_pos = data["t_pos"]
        t_seq = data["t_seq"]

        # for VE diffusion
        if c.diffusion_kind == "edm":
            pos_noised = diffuse_coordinates_edm(
                hk.next_rng_key(), pos, batch, t_pos[:, None]
            )
            seq_noised, corrupt_aa = diffuse_sequence(hk.next_rng_key(), seq, t_seq)
        # for VP, VP-scaled and flow-matching
        if c.diffusion_kind in ("vp", "vpfixed", "flow"):
            cloud_std = None
            if c.diffusion_kind in ("vpfixed", "flow"):
                cloud_std = c.sigma_data
            if "cloud_std" in data:
                cloud_std = data["cloud_std"]
            pos_noised = diffuse_atom_cloud(
                hk.next_rng_key(), pos, mask, batch, t_pos[:, None],
                cloud_std=cloud_std,flow=c.diffusion_kind == "flow")
            seq_noised, corrupt_aa = diffuse_sequence(hk.next_rng_key(), seq, t_seq)
        pos_noised = pos_noised.to_array()

        return dict(
            seq_noised=seq_noised,
            pos_noised=pos_noised,
            t_pos=t_pos,
            t_seq=t_seq,
            corrupt_aa=corrupt_aa,
        )


class StructureDiffusionNoise:
    """Applies VE, VP or VP-scaled noise to inputs."""
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
                hk.next_rng_key(), pos, batch, t_pos[:, None]
            )
            seq_noised, corrupt_aa = diffuse_sequence(hk.next_rng_key(), seq, t_seq)
        if c.diffusion_kind in ("vp", "vpfixed", "flow"):
            cloud_std = None
            if c.diffusion_kind in ("vpfixed", "flow"):
                cloud_std = c.sigma_data
            if "cloud_std" in data:
                cloud_std = data["cloud_std"]
            pos_noised = diffuse_atom_cloud(
                hk.next_rng_key(), pos, mask, batch, t_pos[:, None],
                cloud_std=cloud_std, flow=c.diffusion_kind == "flow")
            seq_noised, corrupt_aa = diffuse_sequence(
                hk.next_rng_key(), seq, t_seq)
        pos_noised = pos_noised.to_array()

        return dict(
            seq_noised=seq_noised,
            pos_noised=pos_noised,
            t_pos=t_pos,
            t_seq=t_seq,
            corrupt_aa=corrupt_aa,
        )


class StructureDiffusionEncode(StructureDiffusion):
    """Barebones encoder module to use as a component in a composite diffusion pipeline."""
    def __call__(self, data):
        _, pos = self.encode(data)
        return pos


class StructureDiffusionPredict(StructureDiffusionInference):
    """Barebones denoising module to use as a component in a composite diffusion pipeline."""
    def __call__(self, data, prev):
        c = self.config
        diffusion = Diffusion(c)

        # prepare condition / task information
        data.update(self.prepare_condition(data))

        def apply_model(prev):
            result = diffusion(data, prev, predict_aa=True)
            prev = dict(pos=result["pos"], local=result["local"])
            return result, prev

        if prev is None:
            prev = diffusion.init_prev(data)
        result, prev = apply_model(prev)
        return result, prev


def get_sigma_edm(batch, meanval=None, stdval=None,
                  minval=0.01, maxval=80.0, randomness=None):
    """Sample random noise levels for VE diffusion.
    
    Args:
        batch: batch index.
        meanval: optional log-mean of a log normal noise level distribution.
        stdval: optional std of a log normal noise level distribution.
        minval: optional minimum of a uniform noise level distribution.
        maxval: optional maximum of a uniform noise level distribution.
        randomness: optional starting noise that can be provided.
    """
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
                hk.next_rng_key(), (size,), minval=0, maxval=1
            )[batch]
        sigma = minval + randomness * (maxval - minval)
    return sigma, randomness


class Encoder(hk.Module):
    """Salad protein backbone encoder."""
    def __init__(self, config, name: Optional[str] = "encoder"):
        super().__init__(name)
        self.config = config

    def prepare_features(self, data):
        """Prepare input features from a batch of data."""
        c = self.config
        positions = data["pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        pair_mask = (batch[:, None] == batch[None, :]) * mask[:, None] * mask[None, :]
        positions = Vec3Array.from_array(positions)
        frames, local_positions = extract_aa_frames(positions)

        # compute and concatenate augment_size additional
        # pseudo-atom positions for each residue
        augment = VectorLinear(c.augment_size, initializer=init_linear())(
            local_positions
        )
        augment = vector_mean_norm(augment)
        local_positions = jnp.concatenate(
            (local_positions.to_array()[:, :5], augment.to_array()), axis=-2
        )
        local_positions = Vec3Array.from_array(local_positions)
        positions = frames[:, None].apply_to_point(local_positions)
        
        # precompute residue neighbours
        neighbours = get_spatial_neighbours(c.num_neighbours)(
            positions[:, 4], batch, mask
        )
        _, local_positions = extract_aa_frames(positions)
        dist = local_positions.norm()

        # initialize local residue features
        local_features = [
            local_positions.normalized().to_array().reshape(local_positions.shape[0], -1),
            distance_rbf(dist, 0.0, 22.0, 16).reshape(local_positions.shape[0], -1),
            jnp.log(dist + 1),
        ]
        local = MLP(c.local_size * 4, c.local_size, activation=jax.nn.gelu, bias=False)(
            jnp.concatenate(local_features, axis=-1)
        )

        positions = positions.to_array()
        local = hk.LayerNorm([-1], True, True)(local)
        return local, positions, neighbours, resi, chain, batch, mask

    def __call__(self, data):
        c = self.config
        # run a stack of encoder layers over the input
        local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data)
        local = EncoderStack(c, c.encoder_depth)(
            local, pos, neighbours, resi, chain, batch, mask
        )
        # update pseudo-atom positions from the encoder representation
        frames, local_positions = extract_aa_frames(Vec3Array.from_array(pos))
        augment = local_positions[:, 5:]
        update = MLP(
            2 * local.shape[-1],
            augment.shape[1] * 3,
            activation=jax.nn.gelu,
            final_init=init_linear())(local)
        update = update.reshape(update.shape[0], augment.shape[1], 3)
        update = Vec3Array.from_array(update)
        augment += update
        local_positions = local_positions[:, :5]
        augment = vector_mean_norm(augment)
        # combine augmented features with backbone / atom positions
        local_positions = jnp.concatenate(
            (local_positions.to_array(), augment.to_array()), axis=-2)
        pos = (
            frames[:, None]
            .apply_to_point(Vec3Array.from_array(local_positions))
            .to_array()
        )
        return local, pos


class Atom14Encoder(hk.Module):
    """Salad encoder using atom14 positions for each residue.
    
    EXPERIMENTAL: Not used in the salad manuscript.
    """
    def __init__(self, config, name: Optional[str] = "atom14_encoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        positions = data["pos"]
        atom_mask = data["atom_mask"]
        aa = data["aa_gt"]
        positions = Vec3Array.from_array(positions)
        frames, local_positions = extract_aa_frames(positions)
        local_positions = local_positions.to_array()
        local_positions = jnp.where(
            atom_mask[..., None], local_positions, local_positions[:, 4:5]
        )
        local_positions = Vec3Array.from_array(local_positions)
        dist = local_positions.norm()
        local_features = jnp.concatenate([
            local_positions.normalized().to_array().reshape(local_positions.shape[0], -1),
            distance_rbf(dist, 0.0, 22.0, 16).reshape(local_positions.shape[0], -1),
            jnp.log(dist + 1),
            jax.nn.one_hot(aa, 20)], axis=-1)
        learned_pos = MLP(
            c.local_size * 2,
            3 * 14,
            activation=jax.nn.gelu,
            bias=True,
            final_init="linear",)(local_features)
        learned_pos = learned_pos.reshape(learned_pos.shape[0], 14, 3)
        learned_pos = jnp.where(
            atom_mask[..., None], local_positions.to_array(), learned_pos)
        pos = frames[:, None].apply_to_point(Vec3Array.from_array(learned_pos))
        return pos.to_array()


class EncoderBlock(hk.Module):
    """Basic encoder block."""
    def __init__(self, config, name: Optional[str] = "encoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, neighbours, resi, chain, batch, mask):
        c = self.config
        # decide if we're using ResiDual or PreNorm residual branches
        residual_update, residual_input, local_shape = get_residual_gadgets(
            features, c.resi_dual
        )
        pair_feature_function = encoder_pair_features
        attention = SparseStructureAttention
        # embed pair features
        pair, pair_mask = pair_feature_function(c)(
            Vec3Array.from_array(pos), neighbours, resi, chain, batch, mask)
        # attention / message passing module
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), pos, pair, pair_mask,
                neighbours, resi, chain, batch, mask))
        # local feature update
        features = residual_update(
            features,
            Update(c, global_update=False)(
                residual_input(features), pos,
                chain, batch, mask, condition=None))
        return features


class EncoderStack(hk.Module):
    """Stack of encoder blocks."""
    def __init__(self, config, depth=None, name: Optional[str] = "self_cond_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 2 # defaults to depth 2

    def __call__(self, local, pos, neighbours, resi, chain, batch, mask):
        c = self.config

        def stack_inner(block):
            def _inner(data):
                data = block(c)(data, pos, neighbours,
                                resi, chain, batch, mask)
                return data

            return _inner

        stack = block_stack(self.depth, block_size=1, with_state=False)(
            hk.remat(stack_inner(EncoderBlock)))
        if c.resi_dual:
            incremental = local
            local, incremental = stack((local, incremental))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local = hk.LayerNorm([-1], True, True)(stack(local))
        return local


class AADecoderBlock(hk.Module):
    """Amino acid sequence decoder block."""
    def __init__(self, config, name: Optional[str] = "aa_decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(
        self, aa, features, pos, neighbours, resi, chain, batch, mask, priority=None
    ):
        c = self.config
        # decide if we're using ResiDual or PreNorm residual branches
        residual_update, residual_input, _ = get_residual_gadgets(features, c.resi_dual)
        # embed pair features
        pair, pair_mask = aa_decoder_pair_features(c)(
            Vec3Array.from_array(pos), aa, neighbours,
            resi, chain, batch, mask)
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
            features, EncoderUpdate(c)(
                residual_input(features), pos, chain, batch, mask))
        return features


class AADecoderStack(hk.Module):
    """Amino acid decoder stack."""
    def __init__(self, config, depth=None, name: Optional[str] = "aa_decoder_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 3

    def __call__(self, aa, local, pos, neighbours,
                 resi, chain, batch, mask, priority=None):
        c = self.config

        def stack_inner(block):
            def _inner(data):
                data = block(c)(
                    aa, data, pos, neighbours,
                    resi, chain, batch, mask,
                    priority=priority)
                return data

            return _inner

        stack = block_stack(self.depth, block_size=1, with_state=False)(
            hk.remat(stack_inner(AADecoderBlock)))
        if c.resi_dual:
            incremental = local
            local, incremental = stack((local, incremental))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local = hk.LayerNorm([-1], True, True)(stack(local))
        return local


def edm_scaling(sigma_data, sigma_noise, alpha=1.0, beta=1.0):
    """Compute model input and output scale factors for VE diffusion.
    
    Args:
        sigma_data: standard deviation of the data.
        sigma_noise: standard deviation of the noise.
    Returns:
        Dictionary of input, output, skip, and loss scales
        for VE diffusion.
    """
    sigma_data = jnp.maximum(sigma_data, 1e-3)
    sigma_noise = jnp.maximum(sigma_noise, 1e-3)
    denominator = alpha**2 * sigma_data**2 + beta**2 * sigma_noise**2
    in_scale = 1 / jnp.sqrt(denominator)
    refine_in_scale = 1 / sigma_data
    out_scale = beta * sigma_data * sigma_noise / jnp.sqrt(denominator)
    skip_scale = alpha * sigma_data**2 / denominator
    loss_scale = 1 / out_scale**2
    return {
        "in": in_scale,
        "out": out_scale,
        "skip": skip_scale,
        "refine": refine_in_scale,
        "loss": loss_scale,
    }


def preconditioning_scale_factors(config, t, sigma_data):
    """Compute model input and output scale factors."""
    c = config
    # VE diffusion
    if c.preconditioning in ("edm", "edm_scaled"):
        result = edm_scaling(sigma_data, 1.0, alpha=1.0, beta=t)
    # flow matching
    elif c.preconditioning == "flow":
        result = {
            "in": 1.0,
            "out": sigma_data,
            "skip": 1.0,
            "refine": 1,
            "loss": 1 / jnp.maximum(t, 0.01) ** 2,
        }
    # VP, VP-scaled
    else:
        result = {"in": 1.0, "out": sigma_data, "skip": 1.0, "refine": 1, "loss": 1}
    if not c.refine:
        result["refine"] = 1.0
    return result


class Diffusion(hk.Module):
    """Diffusion model wrapper."""
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config
        self.diffusion_block = DiffusionBlock

    def __call__(self, data, prev, override=None, predict_aa=False):
        c = self.config
        # select a diffusion block.
        # For all default models, this will be DiffusionBlock.
        # For minimal / minimal_timeless, it will be MinimalDiffusionBlock.
        # For motif_vp, it will be MotifAdapterDiffusion.
        # All other options are experimental and have not been used
        # in the salad manuscript.
        if c.atom:
            # diffusion on atom14 residues
            self.diffusion_block = AtomDiffusionBlock
        elif c.minimal:
            # diffusion with minimal features (speed over quality)
            self.diffusion_block = MinimalDiffusionBlock
        elif c.distogram_trajectory:
            # diffusion with per-block distogram (not worth it)
            self.diffusion_block = DistogramDiffusionBlock
        elif c.multi_motif:
            # multi-motif scaffolding models
            self.diffusion_block = MotifAdapterDiffusionBlock
        diffusion_stack_module = DiffusionStack
        if c.preconditioning == "edm_scaled":
            # for VE diffusion (this needs input and output scaling)
            diffusion_stack_module = EDMDiffusionStack
        if c.nonequivariant_dense:
            # experimental: can we get away with dense diffusion
            # if we do not use pair features? (in terms of speed, yes. But it's worse)
            diffusion_stack_module = NonEquivariantDenseDiffusionStack
            self.diffusion_block = NonEquivariantDenseDiffusionBlock
        if c.repeat:
            # can we get away with a ~1.5M parameter model
            # by repeating a single block 6 times?
            # not naively - the resulting model is bad.
            diffusion_stack_module = RepeatDiffusionStack
        diffusion_stack = diffusion_stack_module(c, self.diffusion_block)
        aa_decoder = AADecoder(c)
        dssp_decoder = Linear(3, bias=False, initializer="zeros")
        distogram_decoder = DistogramDecoder(c)
        features = self.prepare_features(data, prev, override=override)
        (
            local, pos, prev_pos, condition, pair_condition,
            t_pos, t_seq, resi, chain, batch, mask,
        ) = features

        # get neighbours for distogram supervision
        pos_gt = Vec3Array.from_array(data["pos"])
        distogram_neighbours = get_random_neighbours(c.fape_neighbours)(
            pos_gt[:, 4], batch, mask
        )
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
            local, pos, trajectory = diffusion_stack(
                local, pos, prev_pos, condition, pair_condition,
                t_pos, resi, chain, batch, mask,
                cyclic_mask=cyclic_mask, sup_neighbours=sup_neighbours)
        else:
            local, pos, trajectory = diffusion_stack(
                local, pos, prev_pos, condition, pair_condition,
                t_pos, resi, chain, batch, mask,
                cyclic_mask=cyclic_mask, sup_neighbours=sup_neighbours)
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
        result["distogram"] = distogram_decoder(
            local, pos, distogram_neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask)
        # generate all-atom positions using predicted
        # side-chain torsion angles:
        aatype = data["aa_gt"]
        if predict_aa:
            if c.sample_aa:
                aatype = aa_decoder.sample(
                    data["aa_gt"], local, pos,
                    resi, chain, batch, mask)
            else:
                aatype = jnp.argmax(aa_logits, axis=-1)
        if "aa_condition" in data:
            aatype = jnp.where(
                data["aa_condition"] != 20,
                data["aa_condition"], aatype)
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
        """Initialize self-conditioning features."""
        c = self.config
        return {
            "pos": 0.0 * jax.random.normal(hk.next_rng_key(), data["pos_noised"].shape),
            "local": jnp.zeros((data["pos_noised"].shape[0], c.local_size), dtype=jnp.float32)
        }

    def prepare_features(self, data, prev, override=None):
        """Prepare input features from a batch of data."""
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
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        condition = data["condition"]
        pair_condition = data["pair_condition"]
        edm_time_embedding = lambda x: jnp.concatenate(
            (jnp.log(x[:, None]) / 4, fourier_time_embedding(jnp.log(x) / 4, size=256)),
            axis=-1)
        vp_time_embedding = lambda x: jnp.concatenate(
            (x[:, None], fourier_time_embedding(x, size=256)), axis=-1)
        if c.diffusion_kind in ("vp", "vpfixed", "chroma", "flow"):
            time_embedding = vp_time_embedding
        else:
            time_embedding = edm_time_embedding
        pos_time_features = time_embedding(t_pos)
        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1),
        ]
        if c.time_embedding:
            local_features = [pos_time_features] + local_features
        if c.diffuse_sequence:
            seq_time_features = time_embedding(t_seq)
            local_features += [seq_time_features, jax.nn.one_hot(seq, 21, axis=-1)]
        local_features.append(hk.LayerNorm([-1], True, True)(prev["local"]))
        local_features = jnp.concatenate(local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
        local = hk.LayerNorm([-1], True, True)(local)
        if c.condition_time_embedding:
            condition += Linear(condition.shape[-1], initializer="zeros", bias=False)(
                time_embedding)
        condition = hk.LayerNorm([-1], True, True)(condition)

        return (local, pos, prev["pos"], condition, pair_condition,
                t_pos, t_seq, resi, chain, batch, mask)

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

        # Amino acid NLL loss
        aa_predict_mask = mask * (data["aa_gt"] != 20) * result["corrupt_aa"]
        aa_predict_mask = jnp.where(data["aa_mask"], 0, aa_predict_mask)
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll
        total += c.aa_weight * aa_nll

        # Amino acid trajectory loss (disabled for all models in the manuscript)
        if "aa_trajectory" in result:
            aa_predict_mask = mask * (data["aa_gt"] != 20)
            aa_trajectory_nll = -(
                result["aa_trajectory"]
                * jax.nn.one_hot(data["aa_gt"][None], 20, axis=-1)
            ).sum(axis=-1)
            aa_trajectory_nll = jnp.where(aa_predict_mask[None], aa_trajectory_nll, 0)
            aa_trajectory_nll = aa_trajectory_nll.sum(axis=-1) / jnp.maximum(
                aa_predict_mask.sum()[None], 1)
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
        base_weight = (
            mask
            / jnp.maximum(index_sum(mask.astype(jnp.float32), batch, mask), 1)
            / (batch.max() + 1))
        # diffusion loss (backbone + pseudo-atoms)
        do_clip = jnp.where(
            jax.random.bernoulli(hk.next_rng_key(), c.p_clip, batch.shape)[batch],
            100.0,
            jnp.inf)
        sigma = data["t_pos"]
        loss_weight = preconditioning_scale_factors(c, sigma, c.sigma_data)["loss"]
        diffusion_weight = base_weight * loss_weight
        pos_gt = data["pos"]
        pos_pred = result["pos"]
        # optionally align prediction to ground-truth (not used in the manuscript)
        if c.kabsch:
            pos_gt = index_align(pos_gt, pos_pred, batch, mask)
        pos_mask = mask[..., None] * jnp.ones_like(pos_gt[..., 0], dtype=jnp.bool_)
        # if diffusion atom14 positions, optionally mask atoms that are not
        # present in the ground-truth structure.
        if c.mask_atom14:
            pos_mask *= data["atom_mask"]
        # compute and weight the clipped denoising loss
        dist2 = ((result["pos"] - pos_gt) ** 2).sum(axis=-1)
        dist2 *= pos_mask
        dist2 = jnp.clip(dist2, 0, do_clip[:, None]).sum(axis=-1) / jnp.maximum(
            pos_mask.sum(axis=-1), 1)
        dist2 = jnp.where(mask, dist2, 0)
        pos_loss = (dist2 * diffusion_weight).sum() / 3
        losses["pos"] = pos_loss
        total += c.pos_weight * pos_loss
        # diffusion (backbone + pseudo-atoms trajectory)
        dist2 = ((result["trajectory"] - pos_gt[None]) ** 2).sum(axis=-1)
        dist2 *= pos_mask
        dist2 = jnp.clip(dist2, 0, do_clip[None, :, None]).sum(axis=-1) / jnp.maximum(
            pos_mask.sum(axis=-1), 1)
        dist2 = jnp.where(mask[None], dist2, 0)
        trajectory_pos_loss = (dist2 * diffusion_weight[None, ...]).sum(axis=1) / 3
        trajectory_pos_loss = trajectory_pos_loss.mean()
        losses["pos_trajectory"] = trajectory_pos_loss
        total += c.trajectory_weight * trajectory_pos_loss
        # diffusion (all-atom, atom14)
        atom_pos_pred = result["atom_pos"]
        atom_pos_gt = data["atom_pos"]
        # optionally align prediction to ground-truth (not used in the manuscript)
        if c.kabsch:
            atom_pos_gt = index_align(atom_pos_gt, atom_pos_pred, batch, mask)
        dist2 = ((atom_pos_gt - result["atom_pos"]) ** 2).sum(axis=-1)
        dist2 = jnp.clip(dist2, 0, do_clip[:, None])
        atom_mask = data["atom_mask"]
        # only apply the loss to late diffusion steps (not used in the manuscript)
        if c.x_late:
            atom_mask *= late_mask[..., None]
        x_loss = (jnp.where(atom_mask, dist2, 0)).sum(axis=-1) / 3
        x_loss /= jnp.maximum(atom_mask.sum(axis=-1), 1)
        x_loss = (x_loss * base_weight).sum()
        losses["x"] = x_loss
        total += c.x_weight * x_loss
        # rotation loss
        gt_frames, _ = extract_aa_frames(Vec3Array.from_array(data["atom_pos"]))
        frames, _ = extract_aa_frames(Vec3Array.from_array(result["atom_pos"]))
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
        fape_traj = fape_traj.sum(axis=-1) / jnp.maximum(
            mask_neighbours.sum(axis=1)[None], 1
        )
        fape_traj = (fape_traj * base_weight).sum(axis=-1)
        losses["fape"] = fape_traj[-1] / 3
        losses["fape_trajectory"] = fape_traj.mean() / 3
        fape_loss = (c.fape_weight * fape_traj[-1] + c.fape_trajectory_weight * fape_traj.mean()) / 3
        total += fape_loss

        # all-atom local neighbourhood FAPE
        atom_mask = data["atom_mask"]
        local_neighbours = get_neighbours(c.local_neighbours)(distance, pair_mask)
        gt_pos_neighbours = Vec3Array.from_array(data["atom_pos"][local_neighbours])
        pos_neighbours = Vec3Array.from_array(result["atom_pos"][local_neighbours])
        mask_neighbours = (local_neighbours != -1)[..., None] * data["atom_mask"][local_neighbours]
        gt_local_positions = gt_frames[:, None, None].apply_inverse_to_point(gt_pos_neighbours)
        local_positions = frames[-1, :, None, None].apply_inverse_to_point(pos_neighbours)
        dist = (local_positions - gt_local_positions).norm()
        dist2 = jnp.clip(dist, 0, 10.0)
        local_loss = (jnp.where(mask_neighbours, dist2, 0)).sum(axis=(-1, -2)) / 3
        local_loss /= jnp.maximum(
            mask_neighbours.astype(jnp.float32).sum(axis=(-1, -2)), 1)
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
        distogram_nll = jnp.where(
            pair_mask, distogram_nll, 0).sum(
            axis=-1
        ) / jnp.maximum(pair_mask.sum(axis=-1), 1)
        distogram_nll = (distogram_nll * base_weight).sum()
        losses["distogram"] = distogram_nll
        total += 0.1 * distogram_nll

        # all-atom soft LDDT from the AlphaFold 3 paper (not used in the manuscript)
        if c.soft_lddt:
            pos_gt = Vec3Array.from_array(data["pos"])
            pos_pred = Vec3Array.from_array(result["trajectory"][-1])
            dist_gt = (pos_gt[:, None, :, None] - pos_gt[neighbours, None, :]).norm()
            dist_pred = (
                pos_pred[:, None, :, None] - pos_pred[neighbours, None, :]
            ).norm()
            dd = abs(dist_gt - dist_pred)
            dd_mask = (
                pair_mask[..., None, None]
                * pos_mask[:, None, :, None]
                * pos_mask[neighbours, None, :]
                * (dist_gt < 15.0)
            )
            # same atom within residue
            same_atom = jnp.eye(pos_gt.shape[-1])[None, None]
            # AND same residue
            same_atom *= (neighbours == axis_index(neighbours, axis=0)[:, None])[
                ..., None, None
            ]
            dd_mask = dd_mask * (1 - same_atom) > 0
            ddt = (
                jax.nn.sigmoid(0.5 - dd)
                + jax.nn.sigmoid(1 - dd)
                + jax.nn.sigmoid(2 - dd)
                + jax.nn.sigmoid(4 - dd)
            )
            atom_lddt = ((ddt / 4) * dd_mask).sum(axis=(-1, -3)) / jnp.maximum(
                dd_mask.sum(axis=(-1, -3)), 1)
            residue_lddt = atom_lddt.mean(axis=-1)
            mean_lddt = (base_weight * residue_lddt).sum()
            losses["lddt"] = mean_lddt
            total += c.local_weight * (1 - mean_lddt)

        # distogram trajectory loss (not used in the manuscript and not worth it)
        if c.distogram_trajectory:
            distogram_gt = jax.lax.stop_gradient(distance_one_hot(distance_gt, bins=16))
            distogram_nll = -(result["distogram_trajectory"] * distogram_gt[None]).sum(axis=-1)
            distogram_nll = jnp.where(pair_mask[None], distogram_nll, 0).sum(axis=-1) / jnp.maximum(
                pair_mask[None].sum(axis=-1), 1)
            distogram_nll = (distogram_nll * base_weight[None]).sum(axis=1)
            distogram_nll = distogram_nll.mean()
            losses["distogram_trajectory"] = distogram_nll
            total += 0.1 * distogram_nll

        # violation loss
        res_mask = data["mask"] * late_mask
        pred_mask = get_atom14_mask(data["aa_gt"]) * res_mask[:, None]
        violation, _ = violation_loss(
            data["aa_gt"],
            data["residue_index"],
            result["atom_pos"],
            pred_mask,
            res_mask,
            clash_overlap_tolerance=1.5,
            violation_tolerance_factor=2.0,
            chain_index=data["chain_index"],
            batch_index=data["batch_index"],
            per_residue=False,
        )
        losses["violation"] = violation.mean()
        total += c.violation_weight * violation.mean()
        return total, losses


def get_angle_positions(aa_gt, local, pos):
    frames, local_positions = extract_aa_frames(Vec3Array.from_array(pos))
    features = [
        local,
        local_positions.to_array().reshape(local_positions.shape[0], -1),
        distance_rbf(local_positions.norm(), 0.0, 10.0, 16).reshape(
            local_positions.shape[0], -1),
        jax.nn.one_hot(aa_gt, 21, axis=-1),
    ]
    raw_angles = MLP(
        local.shape[-1] * 2,
        7 * 2,
        bias=False,
        activation=jax.nn.gelu,
        final_init="linear")(jnp.concatenate(features, axis=-1))

    raw_angles = raw_angles.reshape(-1, 7, 2)
    angles = raw_angles / jnp.sqrt(
        jnp.maximum((raw_angles**2).sum(axis=-1, keepdims=True), 1e-6))
    angle_pos, _ = single_protein_sidechains(aa_gt, frames, angles)
    angle_pos = angle_pos.to_array().reshape(-1, 14, 3)
    angle_pos = jnp.concatenate((pos[..., :4, :], angle_pos[..., 4:, :]), axis=-2)
    return raw_angles, angles, angle_pos


class DistogramDecoder(hk.Module):
    """Light-weight distogram predictor from local features."""
    def __init__(self, config, name: Optional[str] = "distogram_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(
        self, local, pos, neighbours, resi, chain, batch, mask, cyclic_mask=None
    ):
        c = self.config
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pos = Vec3Array.from_array(pos)
        # construct pair features
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
        # predict distogram logits
        distogram = MLP(
            pair.shape[-1] * 2, 64,
            activation=jax.nn.gelu,
            final_init="zeros")(pair)
        return jax.nn.log_softmax(distogram, axis=-1)


class AADecoder(hk.Module):
    """ADM amino acid decoder."""
    def __init__(self, config, name: Optional[str] = "aa_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, local, pos, resi, chain, batch, mask, priority=None):
        c = self.config
        # if trained as an autoregressive diffusion model (ADM)
        # embed known amino acids.
        if c.aa_decoder_kind == "adm":
            local += Linear(local.shape[-1], initializer=init_glorot())(
                jax.nn.one_hot(aa, 21))
        # precompute 32 nearest neighbours
        neighbours = extract_neighbours(num_index=0, num_spatial=32, num_random=0)(
            Vec3Array.from_array(pos), resi, chain, batch, mask)
        # if not restricted to a single linear layer, apply a stack
        # of AADecoderBlocks.
        if not c.linear_aa:
            local = AADecoderStack(c, depth=c.aa_decoder_depth)(
                aa, local, pos, neighbours,
                resi, chain, batch, mask, priority=priority)
            local = hk.LayerNorm([-1], True, True)(local)
        return Linear(20, initializer=init_zeros(), bias=False)(local), local

    def sample(self, aa, local, pos, resi, chain, batch, mask):
        """Sample a sequence from the AADecoder in random order."""
        c = self.config
        aa = 20 * jnp.ones_like(aa)

        def iter(i, carry):
            aa = carry
            logits, _ = self(aa, local, pos, resi, chain, batch, mask)
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
        """Retrieve the argmax sequence from the predicted logits."""
        aa = 20 * jnp.ones_like(aa)
        logits, _ = self(aa, local, pos, resi, chain, batch, mask)
        return jnp.argmax(logits, axis=-1)

    def train(self, aa, local, pos, resi, chain, batch, mask):
        """Apply the AADecoder at training time."""
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
            logits, features = self(aa, local, pos, resi, chain, batch, mask)
            logits = jax.nn.log_softmax(logits, axis=-1)
            return logits, features, corrupt_aa
        if c.aa_decoder_kind == "ar":
            # if we're training a random-order autoregressive
            # decoder (a la Protein MPNN):
            # sample an array of priorities for each amino acid
            # without replacement. Lower priority amino acids
            # will be sampled before higher-priority amino acids.
            priority = jax.random.permutation(hk.next_rng_key(), aa.shape[0])
            # compute amino acid logits
            logits, features = self(
                aa, local, pos,
                resi, chain, batch, mask,
                priority=priority)
            logits = jax.nn.log_softmax(logits)
            return logits, features, jnp.ones_like(mask)
        # otherwise, mask out all positions and
        # independently predict logits for each
        # sequence position
        aa = 20 * jnp.ones_like(aa)
        logits, features = self(aa, local, pos, resi, chain, batch, mask)
        logits = jax.nn.log_softmax(logits)
        return logits, features, jnp.ones_like(mask)


class DiffusionStack(hk.Module):
    """Diffusion model stack."""
    def __init__(self, config, block, name: Optional[str] = "diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, prev_pos, condition, pair_condition,
                 time, resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None,):
        c = self.config

        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the diffusion block
                result = block(c)(
                    features, pos, prev_pos, condition, pair_condition,
                    time, resi, chain, batch, mask,
                    cyclic_mask=cyclic_mask, sup_neighbours=sup_neighbours)
                if c.distogram_trajectory:
                    features, pos, distogram = result
                    return (features, pos), (pos, distogram)
                else:
                    features, pos = result
                    trajectory_output = pos
                    # return features & positions for the next block
                    # and positions to construct a trajectory
                    return (features, pos), trajectory_output

            return _inner

        diffusion_block = self.block
        stack = block_stack(c.diffusion_depth, c.block_size, with_state=True)(
            hk.remat(stack_inner(diffusion_block)))
        if c.resi_dual:
            # handle ResiDual local features
            incremental = local
            ((local, incremental), pos), trajectory = stack(((local, incremental), pos))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            (local, pos), trajectory = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        return local, pos, trajectory


class RepeatDiffusionStack(hk.Module):
    """Diffusion stack repeating a single block with weight sharing."""
    def __init__(self, config, block, name: Optional[str] = "repeat_diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, prev_pos, condition, pair_condition,
                 time, resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None,):
        c = self.config

        block = hk.remat(self.block(c))

        def _inner(data, _):
            features, pos = data
            # run the diffusion block
            result = block(
                features, pos, prev_pos, condition, pair_condition,
                time, resi, chain, batch, mask,
                cyclic_mask=cyclic_mask, sup_neighbours=sup_neighbours)
            if c.distogram_trajectory:
                features, pos, distogram = result
                return (features, pos), (pos, distogram)
            else:
                features, pos = result
                trajectory_output = pos
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), trajectory_output

        stack = lambda x: hk.scan(_inner, x, jnp.arange(c.diffusion_depth))
        if c.resi_dual:
            # handle ResiDual local features
            incremental = local
            ((local, incremental), pos), trajectory = stack(((local, incremental), pos))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            (local, pos), trajectory = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        return local, pos, trajectory


class EDMDiffusionStack(hk.Module):
    """VE Diffusion stack with input and output scaling."""
    def __init__(self, config, block, name: Optional[str] = "diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, prev_pos, condition, pair_condition,
                 time, resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None,):
        c = self.config
        scale = edm_scaling(c.sigma_data, 1.0, alpha=1.0, beta=time)
        init_pos = c.sigma_data * scale["in"][:, None, None] * pos
        pos = pos * scale["skip"][:, None, None]

        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the diffusion block
                result = block(c)(
                    features, pos, prev_pos, condition, pair_condition,
                    time, resi, chain, batch, mask,
                    cyclic_mask=cyclic_mask, sup_neighbours=sup_neighbours,
                    init_pos=init_pos)
                if c.distogram_trajectory:
                    features, pos, distogram = result
                    return (features, pos), (pos, distogram)
                elif c.aa_trajectory:
                    features, pos, aa = result
                    return (features, pos), (pos, aa)
                else:
                    features, pos = result
                    trajectory_output = pos
                    # return features & positions for the next block
                    # and positions to construct a trajectory
                    return (features, pos), trajectory_output

            return _inner

        diffusion_block = self.block
        stack = block_stack(c.diffusion_depth, c.block_size, with_state=True)(
            hk.remat(stack_inner(diffusion_block))
        )
        if c.resi_dual:
            # handle ResiDual local features
            incremental = local
            ((local, incremental), pos), trajectory = stack(((local, incremental), pos))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            (local, pos), trajectory = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        return local, pos, trajectory


class NonEquivariantDenseDiffusionStack(hk.Module):
    """EXPERIMENTAL: non-equivariant, non sparse diffusion stack."""
    def __init__(self, config, block, name: Optional[str] = "diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, prev_pos, condition, pair_condition,
                 time, resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None,):
        c = self.config

        def fourier_resi_embedding(x, size=128):
            val = x[..., None] / 10_000 ** (2 * jnp.arange(size // 2) / size)
            return jnp.concatenate((jnp.sin(val), jnp.cos(val)), axis=-1)

        scale = edm_scaling(c.sigma_data, 1.0, alpha=1.0, beta=time)
        init_pos = scale["in"][:, None, None] * pos
        local += Linear(local.shape[-1], bias=False, initializer=init_linear())(
            init_pos.reshape(local.shape[0], -1))
        pos = pos * scale["skip"][:, None, None]

        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the diffusion block
                result = block(c)(
                    features, pos, prev_pos,
                    time, resi, chain, batch, mask)
                features, pos = result
                trajectory_output = pos
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), trajectory_output

            return _inner

        diffusion_block = self.block
        stack = block_stack(c.diffusion_depth, c.block_size, with_state=True)(
            hk.remat(stack_inner(diffusion_block)))
        if c.resi_dual:
            # handle ResiDual local features
            incremental = local
            ((local, incremental), pos), trajectory = stack(((local, incremental), pos))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            (local, pos), trajectory = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        return local, pos, trajectory


def extract_condition_neighbours(num_index, num_spatial, num_random, num_block):
    """Extract neighbours using pair condition information."""
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
        neighbours = get_random_neighbours(num_block)(
            block_contact_dist, batch, mask, neighbours)
        return neighbours

    return inner


def extract_motif_neighbours(count):
    """Extract neighbours from a motif distance map."""
    def inner(pair_condition):
        # neighbours by residue index
        neighbours = get_neighbours(count)(
            pair_condition["dmap"], pair_condition["dmap_mask"], None
        )
        return neighbours

    return inner


def encoder_pair_features(c):
    """Compute pair features for the Encoder."""
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(
            pair.shape[-1] * 2, pair.shape[-1],
            activation=jax.nn.gelu,
            final_init="linear",)(pair)
        return pair, pair_mask

    return inner


def aa_decoder_pair_features(c):
    """Compute pair features for the AADecoder."""
    def inner(pos, aa, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(
                32, one_hot=True)(resi, chain, batch, neighbours))
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
        pair = MLP(
            pair.shape[-1] * 2, pair.shape[-1],
            activation=jax.nn.gelu,
            final_init="linear",)(pair)
        return pair, pair_mask

    return inner


def se_diffusion_pair_features(c):
    """Compute pair features for a non-equivariant diffusion block."""
    def inner(
        pos, pair_condition, neighbours, resi, chain, batch, mask, cyclic_mask=None
    ):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = jnp.arange(pair_mask.shape[0], dtype=jnp.int32)
        if pair_condition is not None:
            pair_condition = jax.tree_util.tree_map(
                lambda x: x[index[:, None], neighbours], pair_condition)
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(
                32, one_hot=True, cyclic=cyclic_mask is not None
            )(resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
        if pair_condition is not None:
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(
                    pair_condition["dmap_mask"][..., None],
                    distance_rbf(pair_condition["dmap"]),
                    0))
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(
                    pair_condition["flags"].any(axis=-1)[..., None],
                    pair_condition["flags"],
                    0))
        pos = Vec3Array.from_array(pos)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        local_neighbourhood = jnp.concatenate((
            jnp.repeat(pos.to_array()[:, None], neighbours.shape[1], axis=1),
            pos.to_array()[neighbours],), axis=-2) / c.sigma_data

        local_neighbourhood = local_neighbourhood.reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            local_neighbourhood)
        vectors = pos[:, :5]
        dirs: jnp.ndarray = (vectors[:, None, :, None] - vectors[neighbours, None, :]).to_array()
        dirs = dirs.reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(dirs)
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(
            pair.shape[-1] * 2, pair.shape[-1],
            activation=jax.nn.gelu,
            final_init="linear")(pair)
        return pair, pair_mask

    return inner


def dmap_pair_features(c):
    """Compute pair features using distogram information."""
    def inner(pos, dmap, neighbours, resi, chain, batch, mask, cyclic_mask=None):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        pair = Linear(c.pair_size, bias=False)(
            distance_rbf(dmap[index[:, None], neighbours], 0.0, 22.0, 16.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(Vec3Array.from_array(pos), neighbours, d_min=0.0, d_max=22.0))
        pair = hk.LayerNorm([-1], True, True)(pair)
        return MLP(pair.shape[-1], pair.shape[-1], final_init="linear")(pair), pair_mask

    return inner


def dmap_local_update(c):
    """Message passing update using distogram information."""
    pair_features = dmap_pair_features(c)

    def inner(features, pos, dmap, neighbours, resi, chain, batch, mask):
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


def minimal_diffusion_pair_features(c):
    """Pair features for minimal diffusion blocks.
    
    Only uses residue index distance, euclidean distance and rotation features.
    """
    def inner(pos, pair_condition, neighbours,
              resi, chain, batch, mask,
              cyclic_mask=None, add_pos=None):
        if add_pos is None:
            add_pos = []
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = jnp.arange(pair_mask.shape[0], dtype=jnp.int32)
        if pair_condition is not None:
            pair_condition = jax.tree_util.tree_map(
                lambda x: x[index[:, None], neighbours], pair_condition)
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(
                32, one_hot=True, cyclic=cyclic_mask is not None)(
                    resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
        if pair_condition is not None:
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(
                    pair_condition["dmap_mask"][..., None],
                    distance_rbf(pair_condition["dmap"]), 0))
            if c.use_omap:
                pair += Linear(c.pair_size, initializer="linear", bias=False)(
                    jnp.where(
                        pair_condition["dmap_mask"][..., None],
                        pair_condition["omap"], 0))
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(
                    pair_condition["flags"].any(axis=-1)[..., None],
                    pair_condition["flags"], 0))
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


def diffusion_pair_features(c):
    """Pair features for the standard diffusion block."""
    def inner(pos, pair_condition, neighbours,
              resi, chain, batch, mask,
              cyclic_mask=None, init_pos=None,):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = jnp.arange(pair_mask.shape[0], dtype=jnp.int32)
        if pair_condition is not None:
            pair_condition = jax.tree_util.tree_map(
                lambda x: x[index[:, None], neighbours], pair_condition)
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(
                32, one_hot=True, cyclic=cyclic_mask is not None)(
                    resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
        # embed pair conditioning information for all neighbours
        if pair_condition is not None:
            # for multi-motif nodels, embed the distance map mask
            if c.multi_motif:
                pair += Linear(c.pair_size, initializer="linear", bias=True)(
                    pair_condition["dmap_mask"][..., None])
            # RBF-embed the distance map
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(
                    pair_condition["dmap_mask"][..., None],
                    distance_rbf(pair_condition["dmap"]), 0))
            # optionally embed orientation maps (not used in the manuscript)
            if c.use_omap:
                pair += Linear(c.pair_size, initializer="linear", bias=False)(
                    jnp.where(
                        pair_condition["dmap_mask"][..., None],
                        pair_condition["omap"], 0))
            # embed pairwise flags (interacting chains, hotspot, transposed hotspot)
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(
                    pair_condition["flags"].any(axis=-1)[..., None],
                    pair_condition["flags"], 0))
        # compute pair features based on current position
        pos = Vec3Array.from_array(pos)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        # optionally compute pair features based on initial position
        if init_pos is not None:
            pos = Vec3Array.from_array(init_pos)
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
            pair += Linear(c.pair_size, bias=False, initializer="linear")(
                position_rotation_features(pos, neighbours))
        # apply 2-layer MLP
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(
            pair.shape[-1] * 2, pair.shape[-1],
            activation=jax.nn.gelu,
            final_init="linear")(pair)
        return pair, pair_mask

    return inner


class MotifAdapterDiffusionBlock(hk.Module):
    """Diffusion block for multi-motif scaffolding."""
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, prev_pos,
                 condition, pair_condition,
                 time, resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None,
                 init_pos=None):
        c = self.config
        residual_update, residual_input, _ = get_residual_gadgets(features, c.resi_dual)
        # get neighbours based on current structure
        current_neighbours = extract_neighbours(
            num_index=16, num_spatial=16, num_random=32)(
                Vec3Array.from_array(pos), resi, chain, batch, mask)
        # get neighbours based on input motif
        motif_neighbours = extract_motif_neighbours(32)(pair_condition)

        pair_feature_function = diffusion_pair_features
        attention = SparseStructureAttention
        if c.equivariance == "semi_equivariant":
            pair_feature_function = se_diffusion_pair_features
            attention = SemiEquivariantSparseStructureAttention

        # compute motif-neighbour attention first
        pair, pair_mask = pair_feature_function(c)(
            pos, pair_condition, motif_neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask, init_pos=prev_pos)
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), prev_pos / c.sigma_data, pair, pair_mask,
                motif_neighbours, resi, chain, batch, mask))
        pair, pair_mask = pair_feature_function(c)(
            pos, pair_condition, current_neighbours,
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
            pos, local_norm, scale=out_scale, symm=c.symm)
        return features, pos.astype(local_norm.dtype)


class DiffusionBlock(hk.Module):
    """Standard diffusion block."""
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, prev_pos,
                 condition, pair_condition, time,
                 resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None, init_pos=None):
        c = self.config
        residual_update, residual_input, _ = get_residual_gadgets(features, c.resi_dual)
        # neighbours based on current position
        current_neighbours = extract_neighbours(
            num_index=16, num_spatial=16, num_random=32)(
                Vec3Array.from_array(pos), resi, chain, batch, mask)
        # neighbours based on (self-)conditioning
        cond_neighbours = extract_condition_neighbours(
            num_index=16, num_spatial=16, num_random=16, num_block=16)(
                Vec3Array.from_array(prev_pos), pair_condition, resi, chain, batch, mask)

        pair_feature_function = diffusion_pair_features
        attention = SparseStructureAttention
        if c.equivariance == "semi_equivariant":
            pair_feature_function = se_diffusion_pair_features
            attention = SemiEquivariantSparseStructureAttention

        # attention over current positions
        pair, pair_mask = pair_feature_function(c)(
            pos, pair_condition, current_neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask, init_pos=init_pos)
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), pos / c.sigma_data, pair, pair_mask,
                current_neighbours, resi, chain, batch, mask))
        # attention over self-conditioning
        pair, pair_mask = pair_feature_function(c)(
            prev_pos, pair_condition, cond_neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask, init_pos=init_pos)
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), prev_pos / c.sigma_data, pair, pair_mask,
                cond_neighbours, resi, chain, batch, mask))
        # global update of local features
        features = residual_update(
            features,
            Update(c, global_update=True)(
                residual_input(features),
                pos, chain, batch, mask, condition))
        if c.resi_dual:
            local, incremental = features
            local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local_norm = residual_input(features)
        # update positions
        out_scale = preconditioning_scale_factors(c, time[:, None], c.sigma_data)["out"]
        position_update_function = update_positions
        if c.equivariance == "semi_equivariant":
            position_update_function = semiequivariant_update_positions
        pos = position_update_function(pos, local_norm, scale=out_scale, symm=c.symm)
        return features, pos.astype(local_norm.dtype)


class MinimalDiffusionBlock(hk.Module):
    """Minimal diffusion block trading off quality for speed."""
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, prev_pos,
                 condition, pair_condition, time,
                 resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None, init_pos=None):
        c = self.config
        residual_update, residual_input, _ = get_residual_gadgets(
            features, c.resi_dual)
        neighbours = extract_condition_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=16,
            num_block=16)(
                Vec3Array.from_array(pos),
                pair_condition,
                resi, chain, batch, mask)

        pair_feature_function = minimal_diffusion_pair_features
        attention = SparseStructureAttention
        add_pos = [prev_pos]
        if init_pos is not None:
            add_pos = [init_pos] + add_pos
        pair, pair_mask = pair_feature_function(c)(
            pos, pair_condition, neighbours,
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
                residual_input(features), pos,
                chain, batch, mask, condition))
        if c.resi_dual:
            local, incremental = features
            local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local_norm = residual_input(features)
        out_scale = preconditioning_scale_factors(
            c, time[:, None], c.sigma_data)["out"]
        position_update_function = update_positions
        pos = position_update_function(
            pos, local_norm, scale=out_scale, symm=c.symm)
        if c.aa_trajectory:
            aa = Linear(20, bias=False, initializer=init_zeros())(local_norm)
            aa = jax.nn.log_softmax(aa, axis=-1)
            return features, pos.astype(local_norm.dtype), aa
        return features, pos.astype(local_norm.dtype)


class NonEquivariantDenseDiffusionBlock(hk.Module):
    """Diffusion block using non-equivariant features and dense attention.
    
    Not used in the manuscript.
    """
    def __init__(self, config, name="neq_dense_diffn"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, prev_pos, time, resi, chain, batch, mask):
        c = self.config
        residual_update, residual_input, _ = get_residual_gadgets(features, c.resi_dual)
        features = residual_update(
            features,
            DenseNonEquivariantPointAttention(
                c.key_size, c.heads, normalize=False,
                use_pair=c.use_pair is not None)(
                    residual_input(features), pos / c.sigma_data,
                    resi, chain, batch, mask),
        )
        features = residual_update(
            features,
            NonEquivariantUpdate(c)(
                residual_input(features),
                jnp.concatenate((pos, prev_pos), axis=-2) / c.sigma_data))
        point_update = Linear(pos.shape[-2] * 3, bias=False, initializer="linear")(
            residual_input(features)).reshape(*pos.shape)
        out_scale = preconditioning_scale_factors(c, time[:, None], c.sigma_data)["out"]
        pos += out_scale[..., None] * point_update
        return features, pos


class AtomDiffusionBlock(hk.Module):
    """Diffusion block for diffusing masked atom14 format structures.
    
    Not used in the manuscript.
    """
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, prev_pos,
                 condition, pair_condition, time,
                 resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None, init_pos=None):
        c = self.config
        residual_update, residual_input, _ = get_residual_gadgets(features, c.resi_dual)
        neighbours = extract_condition_neighbours(
            num_index=16, num_spatial=16, num_random=16, num_block=16)(
                Vec3Array.from_array(pos), pair_condition, resi, chain, batch, mask)

        pair_feature_function = minimal_diffusion_pair_features
        attention = SparseStructureAttention
        add_pos = [prev_pos]
        if init_pos is not None:
            add_pos = [init_pos] + add_pos
        pair, pair_mask = pair_feature_function(c)(
            pos, pair_condition, neighbours,
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
                residual_input(features), pos,
                chain, batch, mask, condition))
        if c.resi_dual:
            local, incremental = features
            local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local_norm = residual_input(features)
        # update positions
        out_scale = preconditioning_scale_factors(c, time[:, None], c.sigma_data)["out"]
        local_update, pos_update = AllAtomInteraction(c)(
            residual_input(features), pos,
            resi, chain, batch, mask)
        features = residual_update(features, local_update)
        pos = update_positions_precomputed(
            pos, pos_update, scale=out_scale, symm=c.symm)
        return features, pos.astype(local_norm.dtype)


class DistogramDiffusionBlock(hk.Module):
    """Diffusion block using distogram neighbours.
    
    Not used in the manuscript.
    """
    def __init__(self, config, name: Optional[str] = "diffusion_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, prev_pos,
                 condition, pair_condition, time,
                 resi, chain, batch, mask,
                 cyclic_mask=None, sup_neighbours=None):
        c = self.config
        distogram = InnerDistogram(c)
        residual_update, residual_input, _ = get_residual_gadgets(features, c.resi_dual)
        distogram_logits, dmap = distogram(
            residual_input(features), resi, chain, batch, None)
        index = axis_index(sup_neighbours, axis=0)
        distogram_logits = distogram_logits[index[:, None], sup_neighbours]

        dmap_neighbours = extract_dmap_neighbours(count=32)(
            dmap, resi, chain, batch, mask)
        current_neighbours = extract_neighbours(
            num_index=16, num_spatial=16, num_random=32)(
                Vec3Array.from_array(pos), resi, chain, batch, mask)
        cond_neighbours = extract_condition_neighbours(
            num_index=16, num_spatial=16, num_random=16, num_block=16)(
                Vec3Array.from_array(prev_pos), pair_condition,
                resi, chain, batch, mask)

        pair_feature_function = diffusion_pair_features
        attention = SparseStructureAttention
        if c.equivariance == "semi_equivariant":
            pair_feature_function = se_diffusion_pair_features
            attention = SemiEquivariantSparseStructureAttention

        features = residual_update(
            features,
            dmap_local_update(c)(residual_input(features),
                                 pos, dmap, dmap_neighbours,
                                 resi, chain, batch, mask))
        pair, pair_mask = pair_feature_function(c)(
            pos, pair_condition, current_neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask)
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), pos / c.sigma_data, pair, pair_mask,
                current_neighbours, resi, chain, batch, mask))
        pair, pair_mask = pair_feature_function(c)(
            prev_pos, pair_condition, cond_neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask)
        features = residual_update(
            features,
            attention(c)(
                residual_input(features), prev_pos / c.sigma_data, pair, pair_mask,
                cond_neighbours, resi, chain, batch, mask))
        # global update of local features
        features = residual_update(
            features,
            Update(c, global_update=True)(
                residual_input(features),
                pos, chain, batch, mask, condition))
        if c.resi_dual:
            local, incremental = features
            local_norm = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local_norm = residual_input(features)
        out_scale = preconditioning_scale_factors(c, time[:, None], c.sigma_data)["out"]
        position_update_function = update_positions
        if c.equivariance == "semi_equivariant":
            position_update_function = semiequivariant_update_positions
        pos = position_update_function(pos, local_norm, scale=out_scale, symm=c.symm)
        return features, pos.astype(local_norm.dtype), distogram_logits


class AllAtomInteraction(hk.Module):
    def __init__(self, config, name: str | None = "all_atom_interaction"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, resi, chain, batch, mask):
        c = self.config
        atom_size = c.atom_size or 8
        atom_count = pos.shape[1]
        pos = Vec3Array.from_array(pos)
        neighbours = get_spatial_neighbours(5)(pos[:, 1], batch, mask)
        frames, _ = extract_aa_frames(pos)
        atom_left = Linear(atom_count * atom_size, bias=True)(local).reshape(
            local.shape[0], atom_count, atom_size)
        atom_right = Linear(atom_count * atom_size, bias=True)(local).reshape(
            local.shape[0], atom_count, atom_size)
        atoms = frames[:, None].apply_inverse_to_point(pos)
        neighbour_atoms = frames[:, None, None].apply_inverse_to_point(pos[neighbours])
        relative = atoms[:, None, :, None] - neighbour_atoms[:, :, None, :]
        direction = relative.normalized().to_array()
        distance = relative.norm()
        atom_pair = atom_left[:, None, :, None] + atom_right[neighbours, None, :]
        atom_pair += Linear(atom_size, bias=False)(direction)
        atom_pair += Linear(atom_size, bias=False)(
            distance_rbf(distance, 0.0, 10.0, 10))
        atom_pair = hk.LayerNorm([-1], True, True)(atom_pair)
        atom_pair = MLP(
            atom_size * 2, atom_size,
            activation=jax.nn.gelu,
            final_init=init_relu())(atom_pair)
        atom_local = jax.nn.gelu(atom_pair).mean(axis=(1, 3))
        atom_update = Linear(3, bias=False, initializer=init_zeros())(atom_local)
        local_update = Linear(local.shape[-1], bias=False, initializer=init_zeros())(
            atom_local.reshape(local.shape[0], -1))
        return local_update, atom_update


def get_residual_gadgets(features, use_resi_dual=True):
    """Get accessors for residual state and update."""
    residual_update = (lambda x, y: resi_dual(*x, y)) if use_resi_dual else prenorm_skip
    residual_input = resi_dual_input if use_resi_dual else prenorm_input
    local_shape = features[0].shape[-1] if use_resi_dual else features.shape[-1]
    return residual_update, residual_input, local_shape


def make_pair_mask(mask, neighbours):
    """Construct a neighbour mask from a residue mask and a neighbour array."""
    return mask[:, None] * mask[neighbours] * (neighbours != -1)


class Condition(hk.Module):
    """Per-residue condition module."""
    def __init__(self, config, name: Optional[str] = "condition"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, dssp, mask,
                 residue_index, chain_index, batch_index,
                 set_condition=None,):
        # sample masks for sequence, dssp and dssp mean conditioning
        sequence_mask, dssp_mask, dssp_mean_mask = self.get_masks(
            chain_index, batch_index, set_condition=set_condition)

        # embed amino acid sequence condition
        c = self.config
        aa_latent = Linear(c.local_size, initializer=init_glorot())(
            jax.nn.one_hot(aa, 21, axis=-1))
        aa_latent = hk.LayerNorm([-1], True, True)(aa_latent)

        # embed secondary structure condition
        dssp_latent = Linear(c.local_size, initializer=init_glorot())(
            jax.nn.one_hot(dssp, 3, axis=-1))
        dssp_latent = hk.LayerNorm([-1], True, True)(dssp_latent)

        # embed mean secondary structure condition
        dssp_mean = index_mean(
            jax.nn.one_hot(dssp, 3, axis=-1), chain_index, mask[:, None])
        dssp_mean /= jnp.maximum(dssp_mean.sum(axis=-1, keepdims=True), 1e-6)
        if (set_condition is not None) and (set_condition["dssp_mean"] is not None):
            dssp_mean = set_condition["dssp_mean"]
        dssp_mean = Linear(c.local_size, initializer=init_glorot())(dssp_mean)
        dssp_mean = hk.LayerNorm([-1], True, True)(dssp_mean)

        # add up conditioning information where it is not masked
        condition = (
            jnp.where(dssp_mask[..., None], dssp_latent, 0)
            + jnp.where(dssp_mean_mask[..., None], dssp_mean, 0)
            + jnp.where(sequence_mask[..., None], aa_latent, 0)
        )
        return condition, sequence_mask, dssp_mask

    def get_masks(self, chain_index, batch_index, set_condition=None):
        """Sample random conditioning masks."""
        if set_condition is not None:
            sequence_mask = set_condition["aa"] != 20
            sse_mask = set_condition["dssp"] != 3
            sse_mean_mask = set_condition["dssp_mean_mask"]
            return sequence_mask, sse_mask, sse_mean_mask

        def mask_me(batch, p=0.5, p_min=0.2, p_max=1.0):
            p_mask = jax.random.uniform(
                hk.next_rng_key(), shape=batch.shape, minval=p_min, maxval=p_max
            )[batch]
            bare_mask = jax.random.bernoulli(hk.next_rng_key(), p_mask)
            keep_mask = jax.random.bernoulli(hk.next_rng_key(), p=p, shape=batch.shape)[
                batch
            ]
            return bare_mask * keep_mask

        # mask condition with global probability of 50 %
        # and local coverage between 20 % and 80 %
        sequence_mask = mask_me(batch_index, p=0.5)
        sse_mask = mask_me(batch_index, p=0.5)
        sse_mean_mask = mask_me(chain_index, p=0.5, p_min=0.8)
        return sequence_mask, sse_mask, sse_mean_mask


def get_pair_condition(pos, aa, block_adjacency, pos_mask,
                       resi, chain, batch, set_condition=None):
    """Construct pair conditioning information."""
    same_batch = batch[:, None] == batch[None, :]
    same_chain = (chain[:, None] == chain[None, :]) * same_batch
    p = jax.random.uniform(
        hk.next_rng_key(), (batch.shape[0],), minval=0.0, maxval=0.1
    )[batch]
    noise = jax.random.bernoulli(hk.next_rng_key(), p)
    # get pairwise interaction information
    results = interactions(
        pos, block_adjacency, pos_mask,
        resi, chain, batch, block_noise=noise)
    pair_condition = jnp.stack((
        results["chain_contact"],
        results["relative_hotspot"],
        results["block_contact"]), axis=-1)
    # 50% chance to drop basic contact info for any chain
    drop_chain_contact_info = jax.random.bernoulli(
        hk.next_rng_key(), 0.5, (chain.shape[0],)
    )[chain]
    # 50% chance to drop hotspot info for any residue in a chain
    drop_chain_hotspot_info = jax.random.bernoulli(
        hk.next_rng_key(), 0.5, (chain.shape[0],)
    )[chain]
    # 0 - 50% chance to drop each individual hotspot
    p_drop = jax.random.uniform(
        hk.next_rng_key(), (chain.shape[0],), minval=0.0, maxval=0.5
    )[chain]
    drop_hotspot_info = jax.random.bernoulli(hk.next_rng_key(), p_drop)
    # 0 - 80% chance to drop block-contact info for any pair of residues
    p_drop = jax.random.uniform(
        hk.next_rng_key(), (chain.shape[0],), minval=0.0, maxval=0.8
    )[chain]
    drop_block_contact_info = jax.random.bernoulli(hk.next_rng_key(), p_drop)
    chain_contact_mask = (
        drop_chain_contact_info[:, None] * drop_chain_contact_info[None, :]
    )
    relative_hotspot_mask = (
        drop_chain_hotspot_info[:, None] * drop_hotspot_info[None, :]
    )
    relative_hotspot_mask = jnp.where(chain_contact_mask, relative_hotspot_mask, 0)
    block_contact_mask = (
        drop_block_contact_info[:, None] * drop_block_contact_info[None, :]
    )
    block_contact_mask = jnp.where(same_chain, block_contact_mask, 0)
    # combine pair condition information
    pair_mask = jnp.stack(
        (chain_contact_mask, relative_hotspot_mask, block_contact_mask), axis=-1
    )
    return jnp.where(pair_mask, pair_condition, 0)


class NonEquivariantUpdate(hk.Module):
    """Update block using non-equivariant features.
    
    Not used in the manuscript."""
    def __init__(self, config, name="neq_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos):
        c = self.config
        pos = pos.reshape(local.shape[0], -1)
        hidden = jnp.concatenate((local, pos), axis=-1)
        gate = Linear(
            local.shape[-1] * c.factor,
            initializer="relu", bias=False)(hidden)
        hidden = jax.nn.gelu(gate) * Linear(
            local.shape[-1] * c.factor,
            initializer="linear", bias=False)(hidden)
        return Linear(local.shape[-1], bias=False, initializer="zeros")(hidden)


class Update(hk.Module):
    """Default update block used in all diffusion blocks."""
    def __init__(self, config, global_update=True,
                 name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config
        self.global_update = global_update

    def __call__(self, local, pos, chain, batch, mask, condition=None):
        c = self.config
        # for non-equivariant models, directly embed scaled atom positions
        if c.equivariance == "semi_equivariant":
            local_pos = (pos - pos[:, None, 1]).reshape(pos.shape[0], -1)
        # otherwise embed side-chain positions in the local frame
        else:
            _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
            local_pos = local_pos.to_array().reshape(local_pos.shape[0], -1)
        # add conditioning information
        if condition is not None:
            local += Linear(
                local.shape[-1],
                initializer=init_zeros(),
                bias=False)(condition)
        # add side-chain information
        local += MLP(
            local.shape[-1] * 2, local.shape[-1],
            activation=jax.nn.gelu,
            final_init=init_zeros())(local_pos)
        # GeGLU with per chain / per item averaging
        local_update = Linear(
            local.shape[-1] * c.factor,
            initializer=init_linear(),
            bias=False)(local)
        local_gate = jax.nn.gelu(Linear(
            local.shape[-1] * c.factor,
            initializer=init_relu(),
            bias=False)(local))
        # optionally add features averaged per chain / per batch
        if self.global_update:
            chain_gate = jax.nn.gelu(Linear(
                local.shape[-1] * c.factor,
                initializer=init_relu(), bias=False)(local))
            batch_gate = jax.nn.gelu(Linear(
                local.shape[-1] * c.factor,
                initializer=init_relu(), bias=False)(local))
        hidden = local_gate * local_update
        if self.global_update:
            hidden += index_mean(batch_gate * local_update, batch, mask[..., None])
            hidden += index_mean(chain_gate * local_update, chain, mask[..., None])
        result = Linear(local.shape[-1], initializer=init_zeros())(hidden)
        return result


class EncoderUpdate(hk.Module):
    """Update block without conditioning and global averaging."""
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, chain, batch, mask):
        c = self.config
        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        local_pos = local_pos.to_array().reshape(local_pos.shape[0], -1)
        local += MLP(
            local.shape[-1] * 2, local.shape[-1],
            activation=jax.nn.gelu,
            final_init=init_zeros(),)(local_pos)
        local_update = Linear(
            local.shape[-1] * c.factor,
            initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(
            local.shape[-1] * c.factor,
            initializer=init_relu(), bias=False)(local))
        local_local = local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(local_local)
        return result


def update_positions(pos, local_norm, scale=10.0, symm=None):
    """Update atom positions based on current positions and model features.
    
    Args:
        pos: array of atom positions.
        local_norm: normalized model features.
        scale: scale of the position update. Default: 10.0 angstroms.
        symm: optional symmetrizer.
    Returns:
        Updated atom positions.
    """
    frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
    pos_update = scale * Linear(
        pos.shape[-2] * 3,
        initializer=init_zeros(),
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
    """Update positions with a pre-computed position update.

    Args:
        pos: array of atom positions.
        pos_update: precomputed positions update.
        scale: scale of the position update. Default: 10.0 angstroms.
        symm: optional symmetrizer.
    Returns:
        Updated atom positions.
    """
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
    """Compute hotspots and interactions between chains.
    
    Args:
        coords: atom coordinates in atom14 format of shape (N, 3+, 3).
        block_adjacency: block adjacency matrix of secondary structure elements.
        mask: residue mask.
        resi: residue index.
        chain: chain index.
        batch: batch index.
    Returns:
        Dictionary of interaction information containing:
        "hotspot": mask of interacting residues.
        "chain_contact": mask of in-contact chains.
        "relative_hotspot": mask of which chain a hotspot is a hotspot for.
        "block_contact": block adjacency matrix.
    """
    coords = coords[:, 1]
    same_batch = batch[:, None] == batch[None, :]
    same_chain = (chain[:, None] == chain[None, :]) * same_batch
    mask = mask[:, 1]
    pair_mask = mask[:, None] * mask[None, :]
    pair_mask *= same_batch
    distances = jnp.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    distances = jnp.where(pair_mask, distances, jnp.inf)
    contact_interaction = distances <= CONTACT_THRESHOLD

    size = resi.shape[0]
    # which chains are in contact with each other at all?
    chain_contact = (
        jnp.zeros((size // 10, size // 10), dtype=jnp.bool_)
        .at[chain[:, None], chain[None, :]]
        .max(contact_interaction))
    chain_contact = jnp.where(jnp.eye(size // 10, size // 10), 0, chain_contact)
    chain_contact = chain_contact[chain[:, None], chain[None, :]]
    chain_contact = jnp.where(same_batch, chain_contact, 0)
    chain_contact = jnp.where(same_chain, 0, chain_contact)
    # which residues on this chain interact with anything else?
    hotspot = jnp.where(same_chain, 0, contact_interaction).any(axis=1)
    # which residues on another chain are interaction hotspots as seen by this chain?
    # take the union of all contacts for each chain across all positions in all other chains.
    relative_hotspot = (
        jnp.zeros((size // 10, contact_interaction.shape[1]), dtype=jnp.bool_)
        .at[chain, :]
        .max(jnp.where(same_chain, 0, contact_interaction))[chain])
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
