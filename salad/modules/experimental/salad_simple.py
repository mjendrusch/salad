import jax
import jax.numpy as jnp

import haiku as hk

from salad.aflib.model.geometry import Vec3Array, Rot3Array
from salad.modules.basic import Linear, MLP, GatedMLP
from salad.modules.transformer import drop
from salad.modules.utils.geometry import (
    index_sum, index_mean, index_std, index_max, extract_aa_frames,
    distance_one_hot)
from salad.modules.geometric import FlashPointAttention
from salad.modules.noise_schedule_benchmark import (
    StructureDiffusion, Condition, get_sigma_edm, diffuse_coordinates_edm,
    unique_chain, block_stack, preconditioning_scale_factors)
from salad.aflib.model.all_atom_multimer import get_atom14_mask
from salad.modules.utils.dssp import assign_dssp

def dropout(c, data):
    if c.use_dropout:
        return drop(data, is_training=not c.eval)
    return data

class SimpleDiffusion(StructureDiffusion):
    def __init__(self, config, name = "simple_diffusion"):
        super().__init__(config, name)

    def __call__(self, data):
        # setup models
        c = self.config
        diffusion = Diffusion(c)
        # prepare input data and condition
        data.update(self.prepare_data(data))
        data.update(self.prepare_condition(data))
        # apply noise
        data.update(self.apply_diffusion(data))
        result = diffusion(data)
        result["chain_index"] = data["chain_index"]

        # compute losses
        total, losses = diffusion.loss(data, result)
        out_dict = dict(results=result, losses=losses)
        return total, out_dict

    def apply_diffusion(self, data):
        c = self.config
        # get position and sequence information
        batch = data["batch_index"]
        pos = Vec3Array.from_array(data["pos"])
        result = dict()
        if c.diffusion_kind == "edm":
            # when using variance-expanding diffusion
            # sample a noise level for each structure in the batch
            # according to a log-normal distribution
            sigma_pos, _ = get_sigma_edm(
                batch,
                meanval=c.pos_mean_sigma,
                stdval=c.pos_std_sigma,
                minval=c.pos_min_sigma,
                maxval=c.pos_max_sigma,
            )
            t_pos = sigma_pos
            if "t_pos" in data:
                t_pos = data["t_pos"]
            # apply noise to the input structure
            pos_noised = diffuse_coordinates_edm(
                hk.next_rng_key(), pos, batch, sigma_pos[:, None]
            )
            result["pos_noised"] = pos_noised.to_array()
            result["t_pos"] = t_pos
        elif c.diffusion_kind == "flow":
            # when using flow matching
            # sample a noise level using mixture of beta / uniform distributions
            t_beta = jax.random.beta(hk.next_rng_key(), 1.9, 1.0, (pos.shape[0],))
            t_uniform = jax.random.uniform(hk.next_rng_key(), (pos.shape[0],))
            # FIXME: this is was broken and was almost a uniform distribution
            # up to model iteration -33
            t = 1.0 - jnp.where(
                jax.random.bernoulli(hk.next_rng_key(), 0.02, (pos.shape[0],)),
                t_uniform, t_beta)
            t_pos = t[batch]
            delta_t = jax.random.uniform(hk.next_rng_key(), t_pos.shape)
            # r_pos = t_pos - delta_t * (1 - t_pos) # FIXME: how to do this?
            # ensure that self-conditioning has consistent t_pos
            if "t_pos" in data:
                t_pos = data["t_pos"]
            pos = pos.to_array()
            noise = c.sigma_data * jax.random.normal(hk.next_rng_key(), pos.shape)
            pos_noised = (1 - t_pos)[:, None, None] * pos + t_pos[:, None, None] * noise
            pos_noised = Vec3Array.from_array(pos_noised)
            result["pos_noised"] = pos_noised.to_array()
            result["t_pos"] = t_pos
        else:
            raise NotImplementedError(f"Noise schedule not implemented: {c.diffusion_kind}")
        return result

    def prepare_data(self, data):
        c = self.config
        # get raw position data in atom14 format
        pos = data["all_atom_positions"][:, :14]
        atom_mask = data["all_atom_mask"][:, :14]
        chain = data["chain_index"]
        batch = data["batch_index"]

        # uniquify chain ids
        chain = unique_chain(chain, batch)
        mask = data["residue_mask"] * atom_mask[:, :3].all(axis=-1)

        # center missing atoms to CA / SMOL-center
        pos = jnp.where(atom_mask[..., None], pos, pos[:, 1, None])
        # center all positions
        center = index_mean(pos[:, 1], batch, atom_mask[:, 1, None])
        pos = pos - center[:, None]
        # if model is not rotation equivariant, apply a random rotation and translation
        cx = Vec3Array.from_array(jax.random.normal(hk.next_rng_key(), (3,)))
        cy = Vec3Array.from_array(jax.random.normal(hk.next_rng_key(), (3,)))
        rot = Rot3Array.from_two_vectors(cx, cy).to_array()
        pos = jnp.einsum("cd,ijc->ijd", rot, pos)
        # FIXME: keep the darn thing centered
        # pos += 10 * jax.random.normal(hk.next_rng_key(), (3,))

        return dict(
            chain_index=chain, mask=mask, atom_pos=pos, atom_mask=atom_mask,
            all_atom_positions=pos, all_atom_mask=atom_mask, pos=pos)

    def prepare_condition(self, data):
        r"""Prepares conditioning information for training."""
        c = self.config
        aa = data["aa_gt"]
        pos = data["atom_pos"]
        pos_mask = data["atom_mask"]
        mask = pos_mask.any(axis=1)
        batch = data["batch_index"]
        chain = data["chain_index"]
        # assign 3-state secondary structure for the input 3D structure
        # returning the secondary structure (dssp), a block-index (blocks)
        # which groups contiguous secondary structure elements and
        # a block adjacency matrix (block_adjacency) which specifies
        # contacts between secondary structure elements.
        dssp, _, _ = assign_dssp(
            pos, data["batch_index"], pos_mask.any(axis=-1)
        )

        # set up conditioning information
        conditions = []

        def make_mask(index):
            do_it = jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)[index]
            p_it = jax.random.uniform(hk.next_rng_key(), index.shape, minval=0.2, maxval=0.8)[index]
            it_mask = do_it * jax.random.bernoulli(hk.next_rng_key(), p_it)
            return do_it, it_mask

        # include conditioning stochastically with p = 0.5 during training
        use_condition = jax.random.bernoulli(hk.next_rng_key(), 0.5, (aa.shape[0],))
        use_condition = use_condition[batch]
        # amino acid sequence conditioning
        _, aa_mask = make_mask(batch)
        aa_mask *= use_condition
        aa_condition = aa_mask[:, None] * Linear(c.local_size, bias=False)(
            jax.nn.one_hot(jnp.where(aa_mask, aa, 20), 21))
        conditions.append(aa_condition)
        # dssp conditioning
        do_dssp, dssp_mask = make_mask(batch)
        dssp_mask *= use_condition
        dssp_condition = dssp_mask[:, None] * Linear(c.local_size, bias=False)(
            jax.nn.one_hot(jnp.where(dssp_mask, dssp, 3), 4))
        conditions.append(dssp_condition)
        # dssp_mean conditioning
        do_mean = (1 - do_dssp) * jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)[batch]
        do_mean *= use_condition
        segment = make_segments(batch, min_size=20, max_size=200)
        segment_active = do_mean * jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)[segment]
        # FIXME: this should be masked!!
        dssp_mean = segment_active[:, None] * index_mean(jax.nn.one_hot(dssp, 3), segment, mask[:, None])
        dssp_mean = jnp.where(segment_active[:, None], dssp_mean, 0.0)
        segment = segment_active[:, None] * freq_embedding(augment_perm(segment, num_classes=100), size=64, min_freq=1e-4, max_freq=2**12)
        dssp_mean_condition = segment_active[:, None] * GatedMLP(
            2 * c.local_size, c.local_size,
            activation=jax.nn.silu,
            final_init="linear")(jnp.concatenate((
                dssp_mean, segment, 1 - segment_active[:, None]), axis=-1))
        conditions.append(dssp_mean_condition)

        # for multi-motif scaffolding:
        # segment the entire batch and accept segments for conditioning with p=0.5
        segi = make_segments(batch, min_size=10, max_size=50)
        # assign each segment into one of 2 segment groups (0 or 1)
        seg_group = jax.random.randint(hk.next_rng_key(), batch.shape, 0, 2)[segi]
        # drop segments from being used for conditioning with p = 0.5
        seg_active = jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)[segi]
        struc_mask = use_condition[:, None] * seg_active[:, None] * pos_mask
        # backbone only aa
        p_bb = jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)
        struc_mask = jnp.where(p_bb[:, None], struc_mask.at[:, 4:].set(False), struc_mask)
        # tip atom aa
        p_ta = (1 - p_bb) * jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)
        struc_mask = jnp.where(
            p_ta[:, None],
            jax.random.bernoulli(hk.next_rng_key(), 0.5, struc_mask.shape) * struc_mask,
            struc_mask)
        seg_group_embedding = freq_embedding(
            augment_perm(seg_group, 100), size=128, min_freq=0.01, max_freq=2**8)
        motif_pos = pos - index_mean(pos[:, 1], seg_group, seg_active[:, None])[:, None, :]
        motif_pos = augment_affine(motif_pos, seg_group, sigma=0.0)
        motif_pos = jnp.where(struc_mask[..., None], motif_pos, 0)
        motif_features = jnp.concatenate([
            seg_group_embedding,
            freq_embedding(motif_pos.reshape(motif_pos.shape[0], -1),
                            size=128, min_freq=0.01, max_freq=2**12),
            struc_mask
        ], axis=-1)
        motif = struc_mask.any(axis=1)[:, None] * GatedMLP(
            c.local_size * 2, c.local_size,
            activation=jax.nn.silu, final_init="linear")(motif_features)
        conditions.append(motif)

        condition = Linear(c.local_size, bias=False)(
            jnp.concatenate(conditions, axis=-1))

        result = dict(
            condition=condition, # condition embedding
            dssp_gt=dssp, # ground truth secondary structure
            aa_mask=jnp.where(use_condition, aa_mask, 0), # mask for amino acid identity conditioning
            dssp_mask=jnp.where(use_condition, dssp_mask, 0), # mask for secondary structure conditioning
        )

        return result

def make_segments(batch, min_size=10, max_size=50):
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
    segment_size = jax.random.randint(hk.next_rng_key(), batch.shape, min_size, max_size)
    # initialize iteration state
    seg_pos = 0
    segment = 0
    segi = jnp.zeros_like(batch)
    init = (segi, seg_pos, segment)
    # iterate over residues and return the segment index (segi)
    segi, seg_pos, segment = jax.lax.while_loop(cond, body, init)
    segi = unique_chain(segi, batch)
    return segi

def augment_perm(x, num_classes=100):
    return jax.random.permutation(hk.next_rng_key(), num_classes)[x]

def augment_affine(x, index, sigma=10.0):
    cx = Vec3Array.from_array(jax.random.normal(hk.next_rng_key(), (x.shape[0], 3)))
    cy = Vec3Array.from_array(jax.random.normal(hk.next_rng_key(), (x.shape[0], 3)))
    rot = Rot3Array.from_two_vectors(cx, cy).to_array()[index]
    x = jnp.einsum("icd,ijc->ijd", rot, x)
    x += sigma * jax.random.normal(hk.next_rng_key(), (x.shape[0], 1, 3))[index]
    return x

class SimpleDiffusionPredict(SimpleDiffusion):
    def __init__(self, config, name="simple_diffusion"):
        super().__init__(config, name)

    def __call__(self, data, prev=None):
        # setup models
        c = self.config
        diffusion = Diffusion(c)
        # prepare condition
        data.update(self.prepare_condition(data))
        # denoise
        result = diffusion(data)
        result["chain_index"] = data["chain_index"]
        result["aatype"] = jnp.argmax(result["aa"], axis=-1)
        result["atom_pos"] = result["pos"]
        result["atom_mask"] = get_atom14_mask(result["aatype"])

        return result, prev

    def prepare_condition(self, data):
        r"""Prepares conditioning information for training."""
        c = self.config
        mask = data["mask"]
        batch = data["batch_index"]
        chain = data["chain_index"]
        # assign 3-state secondary structure for the input 3D structure
        # returning the secondary structure (dssp), a block-index (blocks)
        # which groups contiguous secondary structure elements and
        # a block adjacency matrix (block_adjacency) which specifies
        # contacts between secondary structure elements.
        dssp = jnp.full_like(chain, fill_value=3)

        # set up conditioning information
        conditions = []

        def make_mask(index):
            do_it = 0.0 * jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)[index]
            p_it = jax.random.uniform(hk.next_rng_key(), index.shape, minval=0.2, maxval=0.8)[index]
            it_mask = do_it * jax.random.bernoulli(hk.next_rng_key(), p_it)
            return do_it, it_mask

        # include conditioning stochastically with p = 0.5 during training
        use_condition = jnp.zeros_like(jax.random.bernoulli(hk.next_rng_key(), 0.5, (chain.shape[0],)))
        use_condition = use_condition[batch]
        # amino acid sequence conditioning
        _, aa_mask = make_mask(batch)
        aa_mask *= use_condition
        aa_condition = 0.0 * Linear(c.local_size, bias=False)(
            jax.nn.one_hot(jnp.full_like(chain, fill_value=20), 21))
        conditions.append(aa_condition)
        # dssp conditioning
        do_dssp, dssp_mask = make_mask(batch)
        dssp_mask *= use_condition
        dssp_condition = 0.0 * Linear(c.local_size, bias=False)(
            jax.nn.one_hot(jnp.where(dssp_mask, dssp, 3), 4))
        conditions.append(dssp_condition)
        # dssp_mean conditioning
        do_mean = jnp.zeros_like((1 - do_dssp) * jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)[batch])
        do_mean *= use_condition
        dssp_mean = jnp.full((chain.shape[0], 3), fill_value=1/3, dtype=jnp.float32)
        segment = chain
        segment = freq_embedding(augment_perm(segment, num_classes=100), size=64, min_freq=1e-4, max_freq=2**12)
        dssp_mean_condition = 1.0 * GatedMLP(
            2 * c.local_size, c.local_size,
            activation=jax.nn.silu,
            final_init="linear")(jnp.concatenate((
                dssp_mean, segment, jnp.zeros_like(chain)[:, None]), axis=-1))
        conditions.append(dssp_mean_condition)

        # for multi-motif scaffolding:
        # segment the entire batch and accept segments for conditioning with p=0.5
        segi = make_segments(batch, min_size=10, max_size=50)
        # assign each segment into one of 2 segment groups (0 or 1)
        seg_group = jax.random.randint(hk.next_rng_key(), batch.shape, 0, 2)[segi]
        # drop segments from being used for conditioning with p = 0.5
        struc_mask = jnp.zeros((chain.shape[0], 14), dtype=jnp.bool_)
        motif_pos = jnp.zeros((chain.shape[0], 14, 3), dtype=jnp.float32)
        seg_group_embedding = freq_embedding(
            augment_perm(seg_group, 100), size=128, min_freq=0.01, max_freq=2**8)
        motif_features = jnp.concatenate([
            seg_group_embedding,
            freq_embedding(motif_pos.reshape(motif_pos.shape[0], -1),
                            size=128, min_freq=0.01, max_freq=2**12),
            struc_mask
        ], axis=-1)
        motif = 0.0 * GatedMLP(
            c.local_size * 2, c.local_size,
            activation=jax.nn.silu, final_init="linear")(motif_features)
        conditions.append(motif)

        condition = Linear(c.local_size, bias=False)(
            jnp.concatenate(conditions, axis=-1))

        result = dict(
            condition=condition, # condition embedding
        )

        return result

    # def prepare_condition(self, data):
    #     """Prepare condition information."""
    #     # TODO
    #     c = self.config
    #     result = dict()
    #     cond = Condition(c)
    #     # initialize amino acid conditioning
    #     aa = 20 * jnp.ones_like(data["aa_gt"])
    #     if "aa_condition" in data:
    #         aa = data["aa_condition"]
    #     # initialize secondary structure conditioning
    #     dssp = 3 * jnp.ones_like(data["aa_gt"])
    #     if "dssp_condition" in data:
    #         dssp = data["dssp_condition"]
    #     dssp_mean = jnp.zeros((aa.shape[0], 3), dtype=jnp.float32)
    #     dssp_mean_mask = jnp.zeros(aa.shape, dtype=jnp.bool_)
    #     if "dssp_mean" in data:
    #         dssp_mean = jnp.stack([data["dssp_mean"]] * aa.shape[0], axis=0)
    #         dssp_mean_mask = jnp.ones(aa.shape, dtype=jnp.bool_)
    #     # process everything into per-residue condition features
    #     condition, _, _ = cond(
    #         aa,
    #         dssp,
    #         data["mask"],
    #         data["residue_index"],
    #         data["chain_index"],
    #         data["batch_index"],
    #         set_condition=dict(
    #             aa=aa, dssp=dssp, dssp_mean=dssp_mean, dssp_mean_mask=dssp_mean_mask))
    #     result["condition"] = condition

    #     # initialize distance and orientation map conditioning information
    #     dmap = jnp.zeros((aa.shape[0], aa.shape[0]), dtype=jnp.float32)
    #     omap = jnp.zeros((aa.shape[0], aa.shape[0], 9), dtype=jnp.float32)
    #     dmap_mask = jnp.zeros_like(dmap, dtype=jnp.bool_)
    #     if "dmap_mask" in data:
    #         dmap = data["dmap"]
    #         dmap_mask = data["dmap_mask"]
    #         if "omap" in data:
    #             omap = data["omap"]

    #     # FIXME: do we have pairwise condition at all?
    #     # initialize residue pair flags
    #     # FIXME: currently we do not support block-contact conditioning
    #     # or hotspot conditioning.
    #     chain = data["chain_index"]
    #     batch = data["batch_index"]
    #     same_batch = batch[:, None] == batch[None, :]
    #     chain_contacts = chain[:, None] != chain[None, :]
    #     # FIXME: do we need to constrain chain contacts to same batch explicitly?
    #     chain_contacts *= same_batch
    #     if "chain_contacts" in data:
    #         chain_contacts = data["chain_contacts"]
    #     flags = jnp.concatenate(
    #         (
    #             chain_contacts[..., None],
    #             jnp.zeros((chain.shape[0], chain.shape[0], 2)),
    #         ),
    #         axis=-1,
    #     )
    #     pair_condition = dict(
    #         dmap=dmap,
    #         omap=omap,
    #         dmap_mask=dmap_mask,
    #         flags=flags,
    #     )
    #     result["pair_condition"] = pair_condition
    #     return result

def freq_embedding(t, size=128, min_freq=500.0, max_freq=0.01):
    N_freqs = size // 2
    freqs = 2.0 ** jnp.linspace(jnp.log2(min_freq), jnp.log2(max_freq), N_freqs)
    result = t[..., None] * freqs
    return jnp.stack((jnp.sin(result), jnp.cos(result)), axis=-1).reshape(t.shape[0], -1)

class Diffusion(hk.Module):
    """Diffusion model wrapper."""
    def __init__(self, config, name: str | None = "diffusion"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        # TODO
        c = self.config
        diffusion_stack = SimpleDiffusionStack(c, SimpleDiffusionBlock)
        features = self.prepare_features(data)
        (
            local, pos, condition,
            t_pos, resi, chain, batch, mask,
        ) = features
        start_local = local

        # predict
        local, velocity, trajectory = diffusion_stack(
            local, pos, condition,
            t_pos, resi, chain, batch, mask)
        out_pos = pos + t_pos[:, None, None] * velocity
        aa = jax.nn.log_softmax(Linear(
            20, bias=False, initializer="zeros")(
                hk.LayerNorm([-1], True, True)(local)))
        result = dict()
        if c.return_attn:
            trajectory, attention = trajectory
            result["attention"] = attention
            result["attention"]["start_local"] = start_local
        result["local"] = local
        result["aa"] = aa
        result["velocity"] = 10.0 * velocity
        result["pos"] = 10.0 * out_pos
        result["trajectory"] = 10.0 * trajectory
        return result

    def prepare_features(self, data):
        """Prepare input features from a batch of data."""
        c = self.config
        pos = data["pos_noised"]
        t_pos = data["t_pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        condition = data["condition"]
        # set up small molecule pair features? FIXME
        if c.small_molecule:
            bonds = jax.nn.one_hot(data["smol_condition"]["bonds"], 6)
            bond_distance = jax.nn.one_hot(data["smol_condition"]["bond_distance"], c.num_bond_distance)
            pass # TODO

        aug_resi = resi + index_max(-resi, chain, mask) + jax.random.randint(hk.next_rng_key(), (), -500, 500)
        aug_chain = jax.random.permutation(hk.next_rng_key(), 100)[chain]
        aug_batch = jax.random.permutation(hk.next_rng_key(), 100)[batch]

        pos_time_features = freq_embedding(t_pos, size=128, min_freq=0.01, max_freq=2 ** 8)
        pos = pos / 10.0
        local_features = [
            pos.reshape(pos.shape[0], -1),
            freq_embedding(pos.reshape(pos.shape[0], -1), size=128, min_freq=0.01, max_freq=2 ** 12),
            freq_embedding(aug_resi, size=64, min_freq=0.01, max_freq=2**8),
            freq_embedding(aug_chain, size=64, min_freq=0.01, max_freq=2**8),
            # freq_embedding(aug_batch, size=64, min_freq=0.01, max_freq=2**8),
            pos_time_features,
        ]
        # add Delta t = (t - r) embedding
        if c.tim_target:
            local_features.append(freq_embedding(
                data["t_pos"] - data["r_pos"], size=128, min_freq=0.01, max_freq=2 ** 8))
        local_features = jnp.concatenate(local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.silu,
            bias=False,
            final_init="linear")(local_features)
        local = hk.LayerNorm([-1], True, True)(local)
        condition += Linear(condition.shape[-1], initializer="zeros", bias=False)(
            pos_time_features)
        condition = hk.LayerNorm([-1], True, True)(condition)

        return (local, pos, condition,
                t_pos, resi, chain, batch, mask)

    # def prepare_condition(self, data):
    #     pass # TODO

    def loss(self, data, result):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        atom_mask = data["atom_mask"]
        losses = dict()
        total = 0.0

        # Amino acid decoder NLL loss
        aa_predict_mask = mask * (data["aa_gt"] != 20)
        aa_predict_mask = jnp.where(data["aa_mask"], 0, aa_predict_mask)
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll
        total += c.aa_weight * aa_nll

        # sidechain decoder loss
        atom_pos_gt = data["atom_pos"]
        mask_gt = data["atom_mask"]
        # sidechain distance loss
        if c.sidechain_rigid_loss:
            rigid_weight = c.sidechain_rigid_weight if c.sidechain_rigid_weight is not None else 1000
            rigid_loss = _rigid_loss(result["pos"], data, mask, mask_gt)
            losses["rigid"] = rigid_loss
            total += rigid_weight * rigid_loss

        # diffusion losses
        base_weight = (
            mask
            / jnp.maximum(index_sum(mask.astype(jnp.float32), batch, mask), 1)
            / (batch.max() + 1))
        sigma = data["t_pos"]
        loss_weight = preconditioning_scale_factors(c, sigma, c.sigma_data)["loss"]
        diffusion_weight = base_weight# * loss_weight
        pos_gt = data["pos"]
        # p_n = (1 - t) * p_gt + t * n
        # n = (p_n - (1 - t) * p_gt) / t = p_gt + (pn - p_gt) / t
        # p_gt - n = (p_gt - p_n) / t
        # (p_gt - p_n) / t = (p_gt - (1 - t) * p_gt - t * n) / t = (t * p_gt - t * n) / t = p_gt - n
        # v_wrong = p_gt + (p_gt - p_n) / t
        # p_n + t * v_wrong = p_n + p_gt * t + p_gt - p_n = p_gt * t + p_gt = (1 + t) * p_gt
        # FIXME: this is wrong, what was I smoking?
        # velocity_gt = pos_gt - (data["pos_noised"] - pos_gt) / jnp.maximum(1e-6, sigma)[:, None, None]
        velocity_gt = (pos_gt - data["pos_noised"]) / jnp.maximum(1e-6, sigma)[:, None, None]
        trajectory_pred = (data["pos_noised"][None] + sigma[None, :, None, None] * result["trajectory"])
        dist2 = (((pos_gt[None] - trajectory_pred) / jnp.maximum(1e-6, sigma[None, :, None, None])) ** 2).sum(axis=-1)
        # v = result["trajectory"][-1]
        # pos_pred = data["pos_noised"] + sigma[:, None, None] * v
        # pos_pred - pos_gt = pos_noised + t * v - pos_gt = (1 - t) * pos_gt + t * noise + t * v - pos_gt = t * (v - (pos_gt - noise))
        if c.tim_target:
            velocity_gt = velocity_gt + (data["r_pos"] - data["t_pos"])[:, None, None] * result["Dt_velocity"]
        pos_mask = mask[..., None] * jnp.ones_like(pos_gt[..., 0], dtype=jnp.bool_)
        # if diffusion atom14 positions, optionally mask atoms that are not
        # present in the ground-truth structure.
        if c.mask_atom14:
            pos_mask *= data["atom_mask"]
        # diffusion (backbone + pseudo-atoms trajectory)
        # dist2 = ((result["trajectory"] - velocity_gt[None]) ** 2).sum(axis=-1)
        dist2 *= pos_mask
        dist2_unclipped = dist2.sum(axis=-1) / jnp.maximum(
            pos_mask.sum(axis=-1), 1)
        dist2 = dist2_unclipped
        dist2 = jnp.where(mask[None], dist2, 0)
        trajectory_pos_loss = (dist2 * diffusion_weight[None, ...]).sum(axis=1) / 3
        losses["pos"] = trajectory_pos_loss[-1]
        losses["pos_trajectory"] = trajectory_pos_loss.mean()
        total += c.trajectory_weight * trajectory_pos_loss.mean(axis=0) + c.pos_weight * trajectory_pos_loss[-1]
        
        # # distogram loss
        # same_batch = batch[:, None] == batch[None, :]
        # pair_mask = mask[:, None] * mask[:, None]
        # pair_mask = pair_mask * same_batch > 0
        # ca = pos_gt[:, 1]
        # distance_gt = (ca[:, None] - ca[None, :]).norm()
        # distogram_gt = jax.lax.stop_gradient(distance_one_hot(distance_gt))
        # distogram_nll = -(result["distogram"] * distogram_gt).sum(axis=-1)
        # distogram_nll = jnp.where(
        #     pair_mask, distogram_nll, 0).sum(
        #     axis=-1
        # ) / jnp.maximum(pair_mask.sum(axis=-1), 1)
        # distogram_nll = (distogram_nll * base_weight).sum()
        # losses["distogram"] = distogram_nll
        # total += 0.1 * distogram_nll

        return total, losses

def _rigid_loss(atom_pos_pred, data, mask, atom_mask):
    aa_index = jnp.where(mask, data["aa_gt"], 20)
    atom_pos = Vec3Array.from_array(data["atom_pos"])
    atom_dist = (atom_pos[:, :, None] - atom_pos[:, None, :]).norm()
    atom_pos_pred = Vec3Array.from_array(atom_pos_pred)
    atom_dist_pred = (atom_pos_pred[:, :, None] - atom_pos_pred[:, None, :]).norm()
    atom_pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
    dist_mean = index_mean(atom_dist, aa_index, mask=atom_pair_mask)
    dist_std = index_std(atom_dist, aa_index, mask=atom_pair_mask)
    atom_pair_mask = atom_pair_mask * (dist_std < 0.1) > 0
    rigid_loss = jnp.where(atom_pair_mask, ((atom_dist_pred - dist_mean)) ** 2, 0).sum()
    rigid_loss /= jnp.maximum(atom_pair_mask.sum(), 1)
    return rigid_loss

class SimpleDiffusionBlock(hk.Module):
    def __init__(self, config, name = "simple_diffn_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, condition, mask,
                 pair_bias=None, index=None):
        c = self.config
        if c.embed_pred_pos:
            local += MLP(
                local.shape[-1] * 2, local.shape[-1],
                activation=jax.nn.silu,
                final_init="zeros")(freq_embedding(
                    pos, size=128, min_freq=0.01, max_freq=2 ** 8))
        if c.use_attn:
            attention_input = CondNorm()(local, condition)
            if c.add_pos_embedding:
                attention_input += Linear(
                    local.shape[-1], bias=False, initializer="linear")(
                        jnp.concatenate((
                            freq_embedding(pos[:, 1], 64, min_freq=0.01, max_freq=2**8),
                            freq_embedding(index["residue_index"], 64, min_freq=0.01, max_freq=2**8),
                            freq_embedding(index["chain_index"], 64, min_freq=0.01, max_freq=2**8)
                        ), axis=1))
            attn_update = FlashPointAttention(
                    heads=c.heads, key_size=c.key_size,
                    final_init="linear",
                    equivariant=False,
                    return_attn=c.return_attn,
                    overscale_dfactor=False,
                    key_norm_params=False,
                    point_init="zeros")(
                attention_input,
                extract_aa_frames(Vec3Array.from_array(pos))[0].to_array(),
                mask, pair_bias=pair_bias)
            if c.return_attn:
                attn_update, attention = attn_update
                attention["update"] = attn_update
            local += dropout(c, attn_update)
        local += dropout(c, GatedMLP(
            local.shape[-1] * 2,
            local.shape[-1],
            final_init="linear")(CondNorm()(local, condition)))
        local_norm = CondNorm()(local, condition)
        velocity = Linear(14 * 3, initializer="linear", bias=False)(local_norm)
        velocity = velocity.reshape(velocity.shape[0], 14, 3)
        if c.return_attn:
            return local, velocity, attention
        return local, velocity

class SimpleDiffusionStack(hk.Module):
    """VE Diffusion stack with input and output scaling."""
    def __init__(self, config, block, name: str | None = "simple_diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, condition,
                 time, resi, chain, batch, mask,
                 pair_bias=None):
        c = self.config
        index = dict(residue_index=resi, chain_index=chain, batch_index=batch)
        scale = preconditioning_scale_factors(c, time, c.sigma_data)
        # Ensure that positions are centered prior to scaling
        # failing to do so will cause issues during model training
        if isinstance(scale["in"], float):
            pos = pos * scale["in"]
        else:
            pos = pos * scale["in"][:, None, None]
        same_batch = batch[:, None] == batch[None, :]
        pair_bias = jnp.where(same_batch, 0.0, -1e12)[..., None]
        def stack_inner(block):
            def _inner(data):
                local, pred_pos = data
                # run the diffusion block
                result = block(c)(
                    local, pos, condition, mask,
                    pair_bias=pair_bias, index=index)
                if c.return_attn:
                    local, velocity, attention = result
                else:
                    local, velocity = result
                pred_pos = velocity
                trajectory_output = velocity
                # return local & positions for the next block
                # and positions to construct a trajectory
                if c.return_attn:
                    return (local, pred_pos), (trajectory_output, attention)
                return (local, pred_pos), trajectory_output

            return _inner

        diffusion_block = self.block
        if c.repeat:
            base_block = diffusion_block(c)
            diffusion_block = lambda _: base_block
        stack = block_stack(c.diffusion_depth, c.block_size, with_state=True)(
            hk.remat(stack_inner(diffusion_block))
        )
        (local, _), trajectory = stack((local, jnp.zeros_like(pos)))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        if c.return_attn:
            trajectory, attention = trajectory
            return local, trajectory[-1], (trajectory, attention)
        return local, trajectory[-1], trajectory

class CondNorm(hk.Module):
    def __init__(self, name = "cond_norm"):
        super().__init__(name)

    def __call__(self, data, condition):
        data_norm = hk.LayerNorm([-1], False, False)(data)
        scale = jax.nn.sigmoid(Linear(data.shape[-1], initializer="linear")(condition))
        bias = Linear(data.shape[-1], initializer="zeros")(condition)
        return scale * data_norm + bias
