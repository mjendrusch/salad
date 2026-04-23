import jax
import jax.numpy as jnp

import haiku as hk

from salad.aflib.model.geometry import Vec3Array, Rigid3Array, Rot3Array
from salad.modules.basic import Linear, MLP, GatedMLP
from salad.modules.transformer import masked_softmax, drop
from salad.modules.utils.module_ops import stop_parameter_gradient
from salad.modules.utils.geometry import (
    get_spatial_neighbours, axis_index, index_sum, index_max, index_mean, index_std, distance_rbf,
    extract_aa_frames, sequence_relative_position)
from salad.modules.geometric import VectorLinear, SparseInvariantPointAttention, HybridPointAttention
from salad.modules.noise_schedule_benchmark import (
    StructureDiffusion, Condition, get_sigma_edm, diffuse_coordinates_edm, diffuse_atom_cloud,
    unique_chain, update_positions, encoder_pair_features, block_stack,
    preconditioning_scale_factors, distance_one_hot, violation_loss, fourier_time_embedding,
    extract_neighbours, distance_features, direction_features, position_rotation_features,
    pair_vector_features)
from salad.aflib.model.all_atom_multimer import get_atom14_mask, atom37_to_atom14, get_atom37_mask, atom14_to_atom37

def dropout(c, data):
    if c.use_dropout:
        return drop(data, is_training=not c.eval)
    return data

class HybridDiffusion(StructureDiffusion):
    def __init__(self, config, name = "hybrid_diffusion"):
        super().__init__(config, name)

    def __call__(self, data):
        # setup models
        c = self.config
        diffusion = Diffusion(c)
        # setup denoising iteration
        def iteration_body(i, prev):
            override = self.apply_diffusion(data)
            if c.no_override:
                override = None
            result = diffusion(data, prev, override=override)
            prev = dict(pos=result["pos"], local=result["local"])
            if c.self_condition_masked:
                prev["pos_mask"] = jnp.ones((prev["pos"].shape[0],), dtype=jnp.bool_)
            if c.self_condition_decoder:
                decoder_result = diffusion.decoder(
                    data, latent_pos=result["pos"], rollout=True)
                prev["local"] = decoder_result["local"]
                prev["aa"] = jax.nn.softmax(decoder_result["aa"], axis=-1)
            return jax.lax.stop_gradient(prev)
        # prepare input data and condition
        data.update(self.prepare_data(data))
        data.update(self.prepare_condition(data))
        # encode / decode
        latent_pos = diffusion.encoder(data)
        if c.encode_latent:
            latent_pos, latent_seq, seq_mu, seq_log_sigma = latent_pos
            data["latent"] = latent_seq
            data["latent_mu"] = seq_mu
            data["latent_log_sigma"] = seq_log_sigma
        data["pos"] = latent_pos
        decoder_result = diffusion.decoder(data, rollout=False)
        # apply noise
        data.update(self.apply_diffusion(data))
        # denoise, randomly using self-conditioning
        prev = diffusion.init_prev(data)
        if not hk.running_init():
            count = jax.random.randint(hk.next_rng_key(), (), 0, 2)
            prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = diffusion(data, prev)
        result["chain_index"] = data["chain_index"]
        result["latent_pos"] = latent_pos
        result["decoder"] = decoder_result
        # run the decoder on the denoised structure
        optional_stop_gradient = stop_parameter_gradient
        if not c.decouple_ae:
            if c.no_decoder_stop:
                optional_stop_gradient = lambda x: x
            latent_seq = None
            if c.encode_latent:
                latent_seq = result["latent"]
            result["denoised_decoder"] = optional_stop_gradient(diffusion.decoder)(
                data, latent_pos=result["pos"], latent_seq=latent_seq, rollout=True)
        
        # compute losses
        total, losses = diffusion.loss(data, result)
        out_dict = dict(results=result, losses=losses)
        return total, out_dict

    def apply_diffusion(self, data):
        c = self.config
        # get position and sequence information
        batch = data["batch_index"]
        pos = Vec3Array.from_array(data["pos"])
        if c.stop_encoder:
            pos = jax.lax.stop_gradient(pos)
        if c.encode_latent:
            seq = data["latent"]
            if True:#c.stop_encoder:
                seq = jax.lax.stop_gradient(seq)
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
            uniform_weight = 0.02 if c.uniform_weight is None else c.uniform_weight
            t = 1.0 - jnp.where(
                jax.random.bernoulli(hk.next_rng_key(), uniform_weight, (pos.shape[0],)),
                t_uniform, t_beta)
            t_pos = t[batch]
            # ensure that self-conditioning has consistent t_pos
            if "t_pos" in data:
                t_pos = data["t_pos"]
            pos = pos.to_array()
            noise = c.sigma_data * jax.random.normal(hk.next_rng_key(), pos.shape)
            pos_noised = (1 - t_pos)[:, None, None] * pos + t_pos[:, None, None] * noise
            pos_noised = Vec3Array.from_array(pos_noised)
            result["pos_noised"] = pos_noised.to_array()
            result["t_pos"] = t_pos
            if c.encode_latent:
                t_beta = jax.random.beta(hk.next_rng_key(), 1.0, 1.5, (pos.shape[0],))
                t_uniform = jax.random.uniform(hk.next_rng_key(), (pos.shape[0],))
                # FIXME: laprot paper has typo relative to code
                # code uses 20% chance for uniform, paper writes 2% chance
                t_seq = 1.0 - jnp.where(
                    jax.random.bernoulli(hk.next_rng_key(), 0.2, (pos.shape[0])),
                    t_uniform, t_beta
                )[batch]
                noise = jax.random.normal(hk.next_rng_key(), seq.shape)
                seq_noised = (1 - t_seq)[:, None] * seq + t_seq[:, None] * noise
                result["latent_noised"] = seq_noised
                result["t_seq"] = t_seq
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
        # if model is not rotation equivariant, apply a random rotation
        if c.non_equivariant:
            cx = Vec3Array.from_array(jax.random.normal(hk.next_rng_key(), (3,)))
            cy = Vec3Array.from_array(jax.random.normal(hk.next_rng_key(), (3,)))
            rot = Rot3Array.from_two_vectors(cx, cy).to_array()
            pos = jnp.einsum("cd,ijc->ijd", rot, pos)

        return dict(
            chain_index=chain, mask=mask, atom_pos=pos, atom_mask=atom_mask,
            all_atom_positions=pos, all_atom_mask=atom_mask, pos=pos)

class HybridDiffusionPredict(HybridDiffusion):
    def __init__(self, config, name="hybrid_diffusion"):
        super().__init__(config, name)

    def __call__(self, data, prev):
        # setup models
        c = self.config
        diffusion = Diffusion(c)
        # prepare condition
        data.update(self.prepare_condition(data))
        # # encode / decode
        # latent_pos = diffusion.encoder(data)
        # data["pos"] = latent_pos
        # decoder_result = diffusion.decoder(data)
        # denoise
        result = diffusion(data, prev)
        result["chain_index"] = data["chain_index"]
        # result["latent_pos"] = latent_pos
        # result["decoder"] = decoder_result
        # decode
        latent_pos = result["pos"]
        latent_seq = None
        if c.encode_latent:
            latent_seq = result["latent"]
        decoder_result = diffusion.decoder(
            data, latent_pos=latent_pos, latent_seq=latent_seq, rollout=True)
        result["aa"] = decoder_result["aa"]
        result["aatype"] = jnp.argmax(result["aa"], axis=-1)
        result["atom_pos"] = decoder_result["atom_pos"]
        result["atom_mask"] = decoder_result["atom_mask"]

        # jax.debug.print("Step t: {t:.2f}", t=data["t_pos"][0])

        new_prev = dict(pos=result["pos"], local=result["local"])
        if "pos_mask" in prev:
            # FIXME: this does strange things
            new_prev["pos_mask"] = jnp.ones_like(prev["pos_mask"])
        if c.self_condition_decoder:
            t = data["t_pos"][0]
            new_prev["aa"] = jnp.where(
                t < 0.3, jax.nn.softmax(decoder_result["aa"], axis=-1), jnp.zeros_like(decoder_result["aa"]))

        return result, new_prev

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
        batch = data["batch_index"]
        same_batch = batch[:, None] == batch[None, :]
        chain_contacts = chain[:, None] != chain[None, :]
        # FIXME: do we need to constrain chain contacts to same batch explicitly?
        chain_contacts *= same_batch
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


class Decoder(hk.Module):
    def __init__(self, config, name = "decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data, latent_pos=None, latent_seq=None, rollout=False):
        c = self.config
        local, latent_pos, aatype, neighbours, resi, chain, batch, update_mask, mask = self.prepare_features(
            data, latent_pos=latent_pos, latent_seq=latent_seq)
        local, latent_pos = EncoderStack(c, depth=c.decoder_depth or 2)(
            local, latent_pos, neighbours, resi, chain, batch, update_mask, mask)
        local = hk.LayerNorm([-1], True, True)(local)
        result = dict()
        result["aa"] = jax.nn.log_softmax(
            Linear(20, initializer="zeros", bias=False)(local))
        if c.eval:
            aatype = jnp.argmax(result["aa"], axis=-1)
        result["aatype"] = aatype
        result["local"] = local
        # aatype = jax.nn.one_hot(aatype, 20, axis=-1)
        if c.decoder_diffusion:
            diffusion = PositionDiffusion(c)
            if rollout:
                pos, t, atom14_mask = diffusion(
                    local, latent_pos, aatype, pos_gt=None)
            else:
                pos, t, atom14_mask = diffusion(
                    local, latent_pos, aatype, pos_gt=data["atom_pos"])
                result["t_decoder"] = t
        else:
            pos, atom14_mask = PositionDecoder(c)(local, latent_pos, aatype)
        result["atom_pos"] = pos
        result["atom_mask"] = atom14_mask
        return result

    def prepare_features(self, data, latent_pos=None, latent_seq=None):
        c = self.config
        pos = data["pos"]
        if latent_pos is not None:
            pos = latent_pos
        seq = None
        if c.encode_latent:
            seq = data["latent"]
        if latent_seq is not None:
            seq = latent_seq
        latent_pos = pos
        if not c.eval:
            # add noise to latent positions, keeping CA fixed
            center = latent_pos[:, 1]
            latent_pos += 0.1 * jax.random.normal(
                hk.next_rng_key(), latent_pos.shape)
            latent_pos = latent_pos.at[:, 1].set(center)
        # pos = jnp.repeat(pos[:, 1:2], 14, axis=1)
        aatype = data["aa_gt"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        # allow the decoder to update everything
        update_mask = jnp.ones(latent_pos.shape[:-1], dtype=jnp.bool_)# .at[:, :4].set(0)
        neighbours = get_spatial_neighbours(
            c.num_neighbours)(latent_pos[:, 1], batch, mask)
        # local features and frames
        if c.non_equivariant:
            # only translation equivariance - local positions have the center subtracted
            local_pos = Vec3Array.from_array(latent_pos - latent_pos[:, 1:2])
        else:
            frames, local_pos = extract_aa_frames(Vec3Array.from_array(latent_pos))
        distances = distance_rbf(local_pos.norm(), max_distance=10.0)
        directions = local_pos.normalized().to_array()
        local_features = [
            distances.reshape(resi.shape[0], -1),
            directions.reshape(resi.shape[0], -1),
            0.1 * local_pos.to_array().reshape(resi.shape[0], -1)
        ]
        if seq is not None:
            local_features.append(seq)
        local = MLP(
            c.local_size * 2, c.local_size,
            activation=jax.nn.gelu, final_init="linear")(
                jnp.concatenate(local_features, axis=-1))
        local = hk.LayerNorm([-1], True, True)(local)
        return local, latent_pos, aatype, neighbours, resi, chain, batch, update_mask, mask

    def loss(self, result, data):
        mask = data["mask"]
        atom_mask = data["all_atom_mask"]
        pos_gt = data["all_atom_positions"]
        pos = result["atom_pos"]
        latent_pos = data["pos"]
        aa_gt = data["aa_gt"]
        aa = result["aa"]
        batch = data["batch_index"]
        losses = dict()
        # position L2 loss
        batch_weight = 1 / (batch.max() + 1) / index_sum(
            mask.astype(jnp.int32), batch, mask, apply_mask=False)
        pos_loss = (((pos - pos_gt) ** 2).mean(axis=-1) * atom_mask).sum(axis=1)
        pos_loss /= jnp.maximum(atom_mask.sum(axis=1), 1)
        pos_loss = (pos_loss * batch_weight).sum()
        losses["atom_pos"] = pos_loss
        # sequence NLL loss
        aa_nll = -(aa * jax.nn.one_hot(aa_gt, 20, axis=-1)).sum(axis=-1)
        aa_nll = (aa_nll * mask).sum() / jnp.maximum(1, mask.sum())
        losses["aa"] = aa_nll
        # latent pos regularization
        regularization = Vec3Array.from_array(latent_pos - latent_pos[:, 1:2]).norm2()
        regularization = (regularization * mask).sum() / jnp.maximum(1, mask.sum())
        losses["pos_regularization"] = regularization
        # total losses
        # FIXME: 1e-3 changed to 1e-2
        total = losses["aa"] + 10 * losses["atom_pos"] + 1e-2 * losses["pos_regularization"]
        return total, losses

class PositionDiffusion(hk.Module):
    def __init__(self, config, name = "position_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, latent_pos, aatype, pos_gt=None):
        c = self.config
        num_atom = 14
        if c.decoder_atom37:
            num_atom = 37
        # setup
        if c.non_equivariant:
            local_latent = Vec3Array.from_array(latent_pos - latent_pos[:, 1:2])
        else:
            frames, local_latent = extract_aa_frames(Vec3Array.from_array(latent_pos))
        local = MLP(2 * local.shape[-1], local.shape[-1], activation=jax.nn.gelu)(jnp.concatenate((
            jax.nn.one_hot(aatype, 20),
            get_atom14_mask(aatype),
            local_latent.norm2(),
            local_latent.normalized().to_array().reshape(local_latent.shape[0], -1),
            hk.LayerNorm([-1], True, True)(local)
        ), axis=-1))
        local_latent = local_latent.to_array()
        local = hk.LayerNorm([-1], True, True)(local)
        # get atom14 mask
        atom_mask = get_atom14_mask(aatype)

        # iteration
        update_mlp = GatedMLP(local.shape[-1] * 2, num_atom * 3, final_init="zeros")
        def apply(pos_noised):
            update = update_mlp(jnp.concatenate((
                local, pos_noised.reshape(pos_noised.shape[0], -1)), axis=-1))
            update = update.reshape(-1, num_atom, 3)
            pos_noised += update
            return pos_noised
        def rollout(denoised, count):
            for idx in range(count):
                t = 1 - idx / count
                noise = jax.random.normal(hk.next_rng_key(), denoised.shape)
                noised = (1 - t) * denoised + t * noise
                denoised = apply(noised)
            return denoised
        t = jax.random.uniform(hk.next_rng_key(), (local.shape[0],))
        if pos_gt is not None:
            _, pos_gt = extract_aa_frames(Vec3Array.from_array(pos_gt))
            pos_gt = pos_gt.to_array()
            noise = jax.random.normal(hk.next_rng_key(), pos_gt.shape)
            noised = (1 - t)[:, None, None] * pos_gt + t[:, None, None] * noise
            pos = apply(noised)
        else:
            noise = jax.random.normal(hk.next_rng_key(), (local.shape[0], num_atom, 3))
            pos = rollout(noise, count=c.num_rollout or 5)
        if c.non_equivariant:
            pos += latent_pos[:, 1:2]
        else:
            pos = (frames[:, None]
                   .apply_to_point(Vec3Array.from_array(pos.reshape(-1, num_atom, 3)))
                   .to_array())
        return pos, t, atom_mask

class PositionDecoder(hk.Module):
    def __init__(self, config, name = "position_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, latent_pos, aatype):
        c = self.config
        num_atom = 14
        if c.decoder_atom37:
            num_atom = 37
        # setup
        if c.non_equivariant:
            local_latent = Vec3Array.from_array(latent_pos - latent_pos[:, 1:2])
        else:
            frames, local_latent = extract_aa_frames(Vec3Array.from_array(latent_pos))
        local = MLP(2 * local.shape[-1], local.shape[-1], activation=jax.nn.gelu)(jnp.concatenate((
            jax.nn.one_hot(aatype, 20),
            get_atom14_mask(aatype),
            local_latent.norm2(),
            local_latent.normalized().to_array().reshape(local_latent.shape[0], -1),
            hk.LayerNorm([-1], True, True)(local)
        ), axis=-1))
        local_latent = local_latent.to_array()
        local = hk.LayerNorm([-1], True, True)(local)
        pos = Linear(num_atom * 3, initializer="linear")(local)
        pos = pos.reshape(pos.shape[0], num_atom, 3)
        if not c.decoder_atom37:
            pos = pos.at[:, :4].set(local_latent[:, :4])
        else:
            index = jnp.array([0, 1, 2, 4, 3], dtype=jnp.int32)
            pos = pos.at[:, index].set(local_latent[:, :5])

        # iteration
        update_mlp = GatedMLP(local.shape[-1] * 2, num_atom * 3, final_init="zeros")
        def body(i, pos):
            update = update_mlp(jnp.concatenate((
                local, pos.reshape(pos.shape[0], -1)), axis=-1))
            update = update.reshape(-1, num_atom, 3)
            # FIXME: allow the decoder to refine backbone atoms 
            pos += update#.at[:, :4].set(local_latent[:, :4])
            return pos
        if hk.running_init():
            pos = body(0, pos)
        else:
            pos = hk.fori_loop(0, 3, body, pos)
        # project to global frame
        if c.non_equivariant:
            pos += latent_pos[:, 1:2]
        else:
            pos = (frames[:, None]
                   .apply_to_point(Vec3Array.from_array(pos.reshape(-1, num_atom, 3)))
                   .to_array())
        # get atom14 mask
        atom_mask = get_atom14_mask(aatype)
        if c.decoder_atom37:
            pos, _ = atom37_to_atom14(
                aatype, Vec3Array.from_array(pos),
                get_atom37_mask(aatype))
            pos = pos.to_array()
            # FIXME: if this changes behaviour, this means that masking is likely wrong in the loss
            pos = jnp.where(atom_mask[..., None], pos, pos[:, 1:2])
        return pos, atom_mask

class DistogramDecoder(hk.Module):
    def __init__(self, config, name = "distogram_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, pair):
        query = Linear(pair.shape[-1], bias=False)(local)
        key = Linear(pair.shape[-1], bias=False)(local)
        ca = Vec3Array.from_array(pos[:, 1])
        dist = distance_rbf((ca[:, None] - ca[None, :]).norm(), bins=16)
        pair += query[:, None] + key[:, None]
        pair += Linear(pair.shape[-1], bias=False)(dist)
        pair = hk.LayerNorm([-1], True, True)(pair)
        return jax.nn.log_softmax(MLP(
            pair.shape[-1] * 2, 64,
            activation=jax.nn.gelu,
            final_init="zeros")(pair))

def latent_update_positions(pos, local, update_mask, frames=None, scale=10.0,
                            non_equivariant=False):
    start_pos = pos
    if non_equivariant:
        local_pos = Vec3Array.from_array(pos - pos[:, 1:2])
    else:
        if frames is None:
            frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        else:
            local_pos = frames[:, None].apply_inverse_to_point(Vec3Array.from_array(pos))
    pos_update = Linear(
        pos.shape[-2] * 3,
        initializer="zeros",
        bias=False)(local)
    pos_update = Vec3Array.from_array(
        pos_update.reshape(*pos_update.shape[:-1], -1, 3))
    local_pos += scale * pos_update

    # project updated pos to global coordinates
    if non_equivariant:
        pos = local_pos.to_array() + pos[:, 1:2]
    else:
        pos = frames[..., None].apply_to_point(local_pos).to_array()
    # keep fixed according to update-mask
    return jnp.where(update_mask[..., None], pos, start_pos)

class Encoder(hk.Module):
    def __init__(self, config, name = "encoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        local, pos, neighbours, resi, chain, batch, update_mask, mask = self.prepare_features(data)
        local, pos = EncoderStack(c)(local, pos, neighbours, resi, chain, batch, update_mask, mask)
        local = hk.LayerNorm([-1], True, True)(local)
        pos = latent_update_positions(pos, local, update_mask, non_equivariant=c.non_equivariant)
        if c.normalize_encoder == "rmsd":
            center = pos[:, 1:2]
            centered = pos - center
            scale = jnp.sqrt(jnp.maximum(centered ** 2, 1e-6).mean(axis=(-1, -2)))
            # FIXME:
            # salad v1 used this normalization:
            # scale = jnp.sqrt(jnp.maximum((centered ** 2).sum(axis=-1), 1e-6).mean(axis=-2))
            # model 22
            # scale_activation = jax.nn.softplus
            # model 23 / 24: 2 * sigmoid
            # model 25 + bias=False
            scale_activation = lambda x: jax.nn.sigmoid(x)
            learned_scale = scale_activation(Linear(
                pos.shape[1], bias=False, initializer="zeros")(local))
            if c.encoder_no_learned_scale:
                learned_scale = jnp.ones_like(learned_scale)
            pos = center + centered / scale[:, None, None] * learned_scale[..., None]
        elif c.normalize_encoder == "length":
            center = pos[:, 1:2]
            centered = pos - center
            centered = Vec3Array.from_array(centered).normalized().to_array()
            scale_activation = lambda x: 3 * jax.nn.sigmoid(x)
            learned_scale = scale_activation(Linear(
                pos.shape[1], bias=False, initializer="zeros")(local))
            pos = center + learned_scale[..., None] * centered 
        if c.encode_latent:
            latent_mu = Linear(8, bias=False, initializer="zeros")(local)
            latent_log_sigma = Linear(8, bias=False, initializer="zeros")(local)
            latent_sigma = jnp.exp(latent_log_sigma)
            latent = latent_mu + latent_sigma * jax.random.normal(hk.next_rng_key(), latent_mu.shape)
            return pos, latent, latent_mu, latent_log_sigma
        return pos

    def prepare_features(self, data):
        # extract variables from data dictionary
        c = self.config
        atom_pos = data["all_atom_positions"]
        center = Vec3Array.from_array(atom_pos[:, 1])
        atom_mask = data["all_atom_mask"]
        mask = atom_mask.any(axis=1)
        # kind = data["residue_type"]
        residue_name = data["aa_gt"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        # prepare protein update mask:
        # do not update center residues, do not update NCaCO for amino acids
        # FIXME: adapt for small molecules
        update_mask = jnp.ones_like(atom_mask)
        if c.update_mask == "bb":
            update_mask = update_mask.at[:, :4].set(0)
        elif c.update_mask == "ca":
            update_mask = update_mask.at[:, 1].set(0)
        # update_mask = jnp.where(
        #     data["is_aa"][:, None], update_mask.at[:4].set(0), update_mask)
        # get amino acid neighbour graph
        neighbours = get_spatial_neighbours(c.num_neighbours)(
            center, batch, mask)
        # potentially also embed relative residue index?
        relative_resi = resi / index_max(resi, chain, mask)
        # set up local features
        # FIXME: change this for backbone-only encoding
        if c.encoder_bb:
            local = LocalNeighbourEmbedding(c)(atom_pos[:, 1], resi, chain, batch, mask)
        else:
            local = MLP(c.local_size * 2, c.local_size, final_init="linear")(
                jnp.concatenate((#jax.nn.one_hot(kind, 3, axis=-1),
                                jax.nn.one_hot(residue_name, 21, axis=-1),
                                atom_mask), axis=-1))
        # fill in missing positions
        local, pos, update_mask = LatentPositions(c)(
            local, atom_pos, neighbours,
            resi, chain, batch,
            update_mask, mask)
        # get frames and local positions
        if c.non_equivariant:
            local_pos = Vec3Array.from_array(pos - pos[:, 1:2])
        else:
            _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        # update local representation with local positions
        directions = local_pos - local_pos[:, 1:2]
        lengths = distance_rbf(directions.norm(), max_distance=8.0)
        local_features = [
            lengths.reshape(lengths.shape[0], -1),
            directions.normalized().to_array().reshape(
                lengths.shape[0], -1),
        ]
        if not c.encoder_bb:
            local_features.append(atom_mask)
        if c.encoder_atom37:
            pos37 = atom14_to_atom37(atom_pos[:, :14], data["aa_gt"])
            mask37 = get_atom37_mask(data["aa_gt"])
            pos37 = jnp.where(mask37[..., None], pos37, pos37[:, 1:2])
            pos37 = Vec3Array.from_array(pos37)
            if c.non_equivariant:
                local37 = pos37 - pos37[:, 1:2]
            else:
                _, local37 = extract_aa_frames(pos37)
            local_features += [
                distance_rbf(local37.norm(), max_distance=6.0, bins=16).reshape(local.shape[0], -1),
                local37.to_array().reshape(local.shape[0], -1), # control the output magnitude
                mask37 # give additional information on existing atoms
            ]
        local += Linear(local.shape[-1], initializer="zeros")(
            jnp.concatenate(local_features, axis=-1))
        return local, pos, neighbours, resi, chain, batch, update_mask, mask

class LocalNeighbourEmbedding(hk.Module):
    def __init__(self, config, name = "local_neighbour_embedding"):
        super().__init__(name)
        self.config = config

    def __call__(self, ca, resi, chain, batch, mask):
        c = self.config
        if not isinstance(ca, Vec3Array):
            ca = Vec3Array.from_array(ca)
        neighbour_count = c.local_neighbour_count or 8
        neighbours = get_spatial_neighbours(neighbour_count)(ca, batch, mask)
        pair_mask = neighbours != -1
        distance = (ca[:, None] - ca[neighbours]).norm()
        resi_distance = jnp.clip(resi[:, None] - resi[neighbours], -32, 32) + 32
        resi_distance = jnp.where(chain[:, None] != chain[neighbours], 65, resi_distance)
        features = jnp.concatenate((
            distance_rbf(distance, bins=64),
            jax.nn.one_hot(resi_distance, 66, axis=-1)
        ), axis=-1)
        message = MLP(c.pair_size * 2, c.local_size, activation=jax.nn.gelu)(features)
        message = jnp.where(pair_mask[..., None], message, 0).sum(axis=1) / neighbour_count
        return hk.LayerNorm([-1], True, True)(message)

class EncoderBlock(hk.Module):
    def __init__(self, config, name = "encoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, neighbours, resi, chain, batch,
                 update_mask, mask):
        c = self.config
        frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        if c.non_equivariant:
            local_pos = Vec3Array.from_array(pos - pos[:, 1:2])
        # position to local
        local += Linear(local.shape[-1], initializer="zeros", bias=False)(
            jnp.concatenate((
                local_pos.normalized().to_array().reshape(local.shape[0], -1),
                distance_rbf(local_pos.norm(),
                             max_distance=10.0,
                             bins=16).reshape(local.shape[0], -1)), axis=-1))
        # attention update
        pair, pair_mask = encoder_pair_features(c)(
            Vec3Array.from_array(pos),
            neighbours, resi, chain, batch, mask)
        pair = dropout(c, pair)
        # pair_mask = (neighbours != -1) * mask[:, None] > 0
        local += dropout(c, SparseInvariantPointAttention(c.key_size, c.heads,
                                               non_equivariant=c.non_equivariant)(
            hk.LayerNorm([-1], True, True)(local),
            pair, frames.to_array(), neighbours, pair_mask))
        # gated feed forward update
        local += dropout(c, GatedMLP(local.shape[-1] * c.factor)(
            hk.LayerNorm([-1], True, True)(local)))
        # position update, keeping CA and SMOL atoms fixed
        pos = latent_update_positions(
            pos, hk.LayerNorm([-1], True, True)(local),
            update_mask=update_mask, scale=1.0,
            non_equivariant=c.non_equivariant)
        return local, pos

class EncoderStack(hk.Module):
    """Stack of encoder blocks."""
    def __init__(self, config, depth = None, name = "encoder_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 2 # defaults to depth 2

    def __call__(self, local, pos, neighbours, resi, chain, batch,
                 update_mask, mask):
        c = self.config

        def stack_inner(block):
            def _inner(data):
                l, p = data
                data = block(c)(l, p, neighbours,
                                resi, chain, batch,
                                update_mask, mask)
                return data

            return _inner

        stack = block_stack(self.depth, block_size=1, with_state=False)(
            hk.remat(stack_inner(EncoderBlock)))
        local, pos = stack((local, pos))
        return local, pos

class LatentPositions(hk.Module):
    def __init__(self, config, name = "latent_positions"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos,
                 neighbours, resi, chain, batch,
                 update_mask, mask):
        c = self.config
        # set up pair mask
        pair_mask = (neighbours != -1) * mask[neighbours]
        # expand positions to augment size
        pos = jnp.concatenate((
            pos, jnp.full((pos.shape[0], c.augment_size, 3), pos[:, 1, None])), axis=1)
        update_mask = jnp.concatenate((
            update_mask, jnp.ones((pos.shape[0], c.augment_size))), axis=1)
        center = Vec3Array.from_array(pos[:, 1])
        # get pair features
        residue_index_info = sequence_relative_position(
            count=64, one_hot=True)(resi, chain, batch, neighbours)
        # get features of central residue
        center_component = Linear(c.pair_size, bias=False)(local)[:, None]
        # get features of neighbours
        neighbour_component = Linear(c.pair_size, bias=False)(local)[neighbours]
        distance_component = Linear(c.pair_size, bias=False)(
            distance_rbf((center[:, None] - center[neighbours]).norm()))
        resi_component = Linear(c.pair_size, bias=False)(residue_index_info)
        # setup pair features
        pair = resi_component + center_component + neighbour_component + distance_component
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(c.pair_size * 2, c.pair_size, final_init="linear", activation=jax.nn.gelu)(pair)
        pair = hk.LayerNorm([-1], True, True)(pair)
        # get neighbour directions
        neighbour_vectors = pos[:, 1][neighbours] - pos[:, 1][:, None]
        # compute multi-head weights
        message_weight = MLP(c.local_size * 2, c.heads, activation=jax.nn.gelu)(pair)
        message_weight = masked_softmax(pair_mask[..., None], message_weight, axis=1)
        message_features = Linear(c.key_size * c.heads)(pair)
        message_features = message_features.reshape(*neighbours.shape, c.heads, c.key_size)
        message = jnp.einsum("ijh,ijhc->ihc", message_weight, message_features)
        # produce position heads
        message_positions = jnp.einsum("ijh,ijd->ihd", message_weight, neighbour_vectors)
        local += Linear(c.local_size, bias=False, initializer="zeros")(message.reshape(local.shape[0], -1))
        local = hk.LayerNorm([-1], True, True)(local)
        # position_gate = jax.nn.sigmoid(Linear(c.augment_size, initializer="linear"))(local)
        position_update = pos[:, 1:2] + VectorLinear(
            pos.shape[1], initializer="linear")(message_positions)
        # positions = pos + position_gate[:, None] * position_update
        positions = jnp.where(update_mask[..., None], position_update, pos)
        return local, positions, update_mask

def _softclip(data):
    cap = 100.0
    softcap = 50.0
    softcapval = softcap / jnp.arcsinh(10.0)
    return jnp.where(data < cap, data, cap + softcapval * jnp.arcsinh((data - cap) / softcapval))

class Diffusion(hk.Module):
    # TODO Fix this
    """Diffusion model wrapper."""
    def __init__(self, config, name: str | None = "diffusion"):
        super().__init__(name)
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def __call__(self, data, prev, override=None, predict_aa=False):
        c = self.config
        if c.staged_diffusion:
            diffusion_stack = StagedDiffusionStack(c, HybridDiffusionBlock, HybridDiffusionBlock)
        else:
            diffusion_stack = DiffusionStack(c, HybridDiffusionBlock)
        distogram_decoder = DistogramDecoder(c)
        if c.encode_latent:
            seq_update = MLP(c.local_size * 2, 8, activation=jax.nn.gelu, final_init="linear")
        features = self.prepare_features(data, prev, override=override)
        (
            local, pos, pair, pair_bias, condition,
            t_pos, resi, chain, batch, mask,
        ) = features

        # predict
        local, pos, trajectory = diffusion_stack(
            local, pos, pair, condition,
            t_pos, resi, chain, batch, mask,
            pair_bias=pair_bias)
        aa_aux = jax.nn.log_softmax(Linear(
            20, bias=False, initializer="zeros")(
                hk.LayerNorm([-1], True, True)(local)))
        # predict & handle sequence, losses etc.
        result = dict()
        if c.encode_latent:
            latent_update = seq_update(hk.LayerNorm([-1], True, True)(local))
            new_latent = data["latent_noised"] + data["t_seq"][:, None] * latent_update
            result["latent"] = new_latent
        result["trajectory"] = trajectory
        result["local"] = local
        result["aa"] = aa_aux
        result["pos"] = pos
        result["distogram"] = distogram_decoder(
            local, pos, pair)
        return result

    def init_prev(self, data):
        """Initialize self-conditioning features."""
        c = self.config
        prev = {
            "pos": 0.0 * jax.random.normal(hk.next_rng_key(), data["pos_noised"].shape),
            "local": jnp.zeros((data["pos_noised"].shape[0], c.local_size), dtype=jnp.float32)
        }
        if c.self_condition_masked:
            prev["pos_mask"] = jnp.zeros((data["pos"].shape[0],), dtype=jnp.bool_)
        if c.self_condition_decoder:
            prev["aa"] = jnp.zeros((data["pos"].shape[0], 20), dtype=jnp.float32)
        return prev

    def prepare_features(self, data, prev, override=None):
        """Prepare input features from a batch of data."""
        c = self.config
        pos = data["pos_noised"]
        seq = data["latent_noised"] if c.encode_latent else None
        t_pos = data["t_pos"]
        t_seq = data["t_seq"] if c.encode_latent else None
        if override is not None:
            pos = override["pos_noised"]
            t_pos = override["t_pos"]
            if c.encode_latent:
                seq = override["latent_noised"]
                t_seq = override["t_seq"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        condition = data["condition"]
        # set up pair condition features for pair feature matrix
        pair_condition = data["pair_condition"]
        pair_condition = jnp.concatenate((
            jnp.where(
                pair_condition["dmap_mask"][..., None],
                distance_rbf(pair_condition["dmap"], bins=16), 0),
            pair_condition["dmap_mask"][..., None].astype(jnp.float32),
            pair_condition["flags"]
        ), axis=-1)
        # set up residue distance features for pair feature matrix
        resi_dist = sequence_relative_position(32, one_hot=True)(resi, chain, batch) # 130
        # set up previous and initial distance features for pair feature matrix
        prev_ca = Vec3Array.from_array(prev["pos"][:, 1])
        prev_dist = distance_rbf((prev_ca[:, None] - prev_ca[None, :]).norm(), bins=16) # 16
        # add masking information to previous positions
        # makes the presence / absence of pos conditioning more clear to the model
        if c.self_condition_masked:
            prev_dist = jnp.where(prev["pos_mask"][:, None, None], prev_dist, 0)
            pos_mask = prev["pos_mask"][:, None, None]
            if c.invert_pos_mask:
                pos_mask = 1 - pos_mask
            prev_dist = jnp.concatenate((
                prev_dist,
                jnp.broadcast_to(pos_mask,
                                 (resi.shape[0], resi.shape[0], 1))), axis=-1)
        scale = preconditioning_scale_factors(c, t_pos, c.sigma_data)
        if isinstance(scale["in"], float):
            init_pos = c.sigma_data * scale["in"] * pos
        else:
            init_pos = c.sigma_data * scale["in"][:, None, None] * pos
        # FIXME: no scaling of inputs for flow!
        if c.diffusion_kind == "flow":
            init_pos = pos
        init_ca = Vec3Array.from_array(init_pos[:, 1])
        init_dist = distance_rbf((init_ca[:, None] - init_ca[None, :]).norm(), bins=16) # 16
        init_pair_features = pair_vector_features(
            Vec3Array.from_array(init_pos), scale=0.1) # 40 * 3 = 120
        # set up small molecule pair features? FIXME
        # small molecules should use full conformer restraints
        # i.e. a full distance matrix and chirality-aware small molecule embedding
        # on top of bond features, etc.
        if c.small_molecule: # TODO
            bonds = jax.nn.one_hot(data["smol_condition"]["bonds"], 6)
            bond_distance = jax.nn.one_hot(data["smol_condition"]["bond_distance"], c.num_bond_distance)
            pass # TODO

        # FIXME: this time embedding makes no sense for t in [0, 1]
        embedding_scale = 100.0
        time_embedding = lambda x: jnp.concatenate(
            (jnp.log(x[:, None]) / 4, fourier_time_embedding(jnp.log(x) / 4, size=256, scale=embedding_scale)),
            axis=-1)
        if c.diffusion_kind != "edm":
            embedding_scale = 1.0
            if True: # FIXME
                embedding_scale = 100.0
            time_embedding = lambda x: fourier_time_embedding(x, size=256, base=10_000, scale=embedding_scale)
        pos_time_features = time_embedding(t_pos)
        seq_time_features = time_embedding(t_seq) if c.encode_latent else None
        pair_features = [
            pair_condition,
            resi_dist,
            prev_dist,
            init_dist,
            init_pair_features,
        ]
        if c.encode_latent:
            seq_left = Linear(c.pair_size, bias=False)(seq)
            seq_right = Linear(c.pair_size, bias=False)(seq)
            seq_pair = (seq_left[:, None] + seq_right[None, :]) / jnp.sqrt(2)
            pair_features.append(seq_pair)
        pair = MLP(c.pair_size * 2, c.pair_size, activation=jax.nn.gelu)(
            jnp.concatenate(pair_features, axis=-1))
        if c.pair_time:
            pair_time = jnp.broadcast_to(
                pos_time_features[:, None, :],
                (resi_dist.shape[0],
                 resi_dist.shape[1],
                 pos_time_features.shape[-1]))
            pair = CondNorm()(
                pair,
                hk.LayerNorm([-1], True, True)(
                    Linear(c.pair_size)(pair_time)))
        else:
            pair = hk.LayerNorm([-1], True, True)(pair)
        pair_bias = Linear(c.heads, initializer="linear")(pair)
        if c.non_equivariant:
            local_pos = Vec3Array.from_array(pos - pos[:, 1:2])
        else:
            _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1),
        ]
        local_features = [pos_time_features] + local_features
        if c.encode_latent: # add in sequence features
            local_features = [seq_time_features, seq] + local_features
        local_features.append(hk.LayerNorm([-1], True, True)(prev["local"]))
        if "aa" in prev:
            prev_aa = jnp.where(jax.random.bernoulli(hk.next_rng_key(), 0.5), prev["aa"], 0)
            local_features.append(prev_aa)
        local_features = jnp.concatenate(local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init="linear")(local_features)
        local = hk.LayerNorm([-1], True, True)(local)
        condition += Linear(condition.shape[-1], initializer="zeros", bias=False)(
            pos_time_features)
        condition = hk.LayerNorm([-1], True, True)(condition)

        return (local, pos, pair, pair_bias, condition,
                t_pos, resi, chain, batch, mask)

    def loss(self, data, result):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        atom_mask = data["atom_mask"]
        losses = dict()
        total = 0.0

        # mask for losses which should only apply late
        # in the diffusion process
        if c.diffusion_kind == "edm":
            late_mask = data["t_pos"] < 5.0
        else:
            late_mask = data["t_pos"] < 0.5
        late_mask *= mask

        # Amino acid decoder NLL loss
        aa_predict_mask = mask * (data["aa_gt"] != 20)
        aa_predict_mask = jnp.where(data["aa_mask"], 0, aa_predict_mask)
        aa_nll = -(result["decoder"]["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll
        total += c.aa_weight * aa_nll

        # auxiliary amino acid loss
        # FIXME: denoised decoder sequence instead of linear head sequence
        if not c.decouple_ae:
            aa_nll = -(result["denoised_decoder"]["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
            aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
            aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
            losses["aa_aux"] = aa_nll
            total += c.aa_weight * aa_nll

        # pseudoatom L2 loss
        # FIXME: explicitly center on the CA position, even if using pseudo-atom latents
        directions = Vec3Array.from_array(data["pos"][:, :] - data["atom_pos"][:, 1:2])
        update_mask = jnp.ones((directions.shape[0], directions.shape[1]), dtype=jnp.bool_)
        if c.update_mask == "bb":
            update_mask = update_mask.at[:, :4].set(0)
        elif c.update_mask == "ca":
            update_mask = update_mask.at[:, 1].set(0)
        pseudoatom_l2 = (directions.norm2() * update_mask).sum(axis=-1) / jnp.maximum(1, update_mask.astype(jnp.float32).sum(axis=-1))
        pseudoatom_l2 = (pseudoatom_l2 * mask).sum() / jnp.maximum(1, mask.astype(jnp.float32).sum())
        losses["regularization"] = pseudoatom_l2
        total += c.regularization_weight * pseudoatom_l2 # FIXME: 0.1 or 0.01?

        # sidechain decoder loss
        atom_pos_gt = data["atom_pos"]
        mask_gt = data["atom_mask"]
        atom_pos_pred = result["decoder"]["atom_pos"]
        sidechain_l2 = (((atom_pos_gt - atom_pos_pred) ** 2).sum(axis=-1) * mask_gt).sum(axis=1)
        sidechain_l2 /= jnp.maximum(1, mask_gt.sum(axis=1))
        # if c.decoder_diffusion:
        #     decoder_time = result["decoder"]["t_decoder"]
        #     sidechain_l2 /= jnp.maximum(decoder_time ** 2, 0.01)
        sidechain_l2 = sidechain_l2.sum() / jnp.maximum(1, mask_gt.any(axis=1).sum())
        losses["sidechain"] = sidechain_l2
        total += c.sidechain_weight * sidechain_l2

        # sidechain distance loss
        if c.sidechain_rigid_loss:
            rigid_weight = c.sidechain_rigid_weight if c.sidechain_rigid_weight is not None else 1000
            rigid_loss = _rigid_loss(result["decoder"]["atom_pos"], data, mask, mask_gt)
            losses["rigid"] = rigid_loss
            denoised_rigid_loss = _rigid_loss(result["denoised_decoder"]["atom_pos"], data, mask, mask_gt)
            losses["denoised_rigid"] = denoised_rigid_loss
            if c.denoised_rigid_loss:
                rigid_loss = (rigid_loss + denoised_rigid_loss) / 2
            total += rigid_weight * rigid_loss
            # aa_index = jnp.where(mask, data["aa_gt"], 20)
            # atom_pos = Vec3Array.from_array(data["atom_pos"])
            # atom_dist = (atom_pos[:, :, None] - atom_pos[:, None, :]).norm()
            # atom_pos_pred = Vec3Array.from_array(atom_pos_pred)
            # atom_dist_pred = (atom_pos_pred[:, :, None] - atom_pos_pred[:, None, :]).norm()
            # atom_pair_mask = mask_gt[:, :, None] * mask_gt[:, None, :]
            # dist_mean = index_mean(atom_dist, aa_index, mask=atom_pair_mask)
            # dist_std = index_std(atom_dist, aa_index, mask=atom_pair_mask)
            # atom_pair_mask = atom_pair_mask * (dist_std < 0.1) > 0
            # rigid_loss = jnp.where(atom_pair_mask, ((atom_dist_pred - dist_mean)) ** 2, 0).sum()
            # rigid_loss /= jnp.maximum(atom_pair_mask.sum(), 1)

        # diffusion losses
        base_weight = (
            mask
            / jnp.maximum(index_sum(mask.astype(jnp.float32), batch, mask), 1)
            / (batch.max() + 1))

        # latent loss
        if c.encode_latent:
            seq_gt = data["latent"]
            seq = result["latent"]
            seq_mu = data["latent_mu"]
            seq_log_sigma = data["latent_log_sigma"]
            seq_sigma = jnp.exp(seq_log_sigma)

            t_seq = data["t_seq"]
            latent_loss = (
                ((seq - jax.lax.stop_gradient(seq_gt)) ** 2).mean(axis=-1)
                  * base_weight / jnp.maximum(t_seq, 0.001) ** 2).sum()
            losses["latent"] = latent_loss
            prior_loss = 0.5 * (seq_sigma ** 2 + seq_mu ** 2 - 1 - 2 * seq_log_sigma).mean(axis=-1)
            prior_loss = jnp.where(mask, prior_loss, 0).sum(axis=0) / jnp.maximum(1, mask.sum())
            losses["prior"] = prior_loss
            total += 0.1 * prior_loss + latent_loss # FIXME: weight latent loss?

        # diffusion loss (backbone + pseudo-atoms)
        do_clip = jnp.where(
            jax.random.bernoulli(hk.next_rng_key(), c.p_clip, batch.shape)[batch],
            100.0,
            jnp.inf)
        clip_val = 10.0 # angstrom
        clipped_weight = c.clipped_weight or 1.0
        unclipped_weight = c.unclipped_weight if c.unclipped_weight is not None else 0

        sigma = data["t_pos"]
        loss_weight = preconditioning_scale_factors(c, sigma, c.sigma_data)["loss"]
        if c.no_loss_weight:
            loss_weight = 1.0
        diffusion_weight = base_weight * loss_weight
        # FIXME: should we stop-gradient this? this is letting gradients leak into the encoder that absolutely shouldn't be there
        # context: pressure to "simplify" sidechain encoding to reduce loss
        # this seems to have a massive impact on the loss, for the better
        pos_gt = data["pos"]
        if c.stop_encoder_loss:
            pos_gt = jax.lax.stop_gradient(pos_gt)
        pos_mask = mask[..., None] * jnp.ones_like(pos_gt[..., 0], dtype=jnp.bool_)
        # if diffusion atom14 positions, optionally mask atoms that are not
        # present in the ground-truth structure.
        if c.mask_atom14:
            pos_mask *= data["atom_mask"]
        # diffusion (backbone + pseudo-atoms trajectory)
        dist2 = ((result["trajectory"] - pos_gt[None]) ** 2).sum(axis=-1)
        dist2 *= pos_mask
        dist2_clipped = jnp.clip(dist2, 0, clip_val ** 2).sum(axis=-1) / jnp.maximum(
            pos_mask.sum(axis=-1), 1)
        dist2_unclipped = dist2.sum(axis=-1) / jnp.maximum(
            pos_mask.sum(axis=-1), 1)
        dist2 = clipped_weight * dist2_clipped + unclipped_weight * dist2_unclipped
        if c.softclip:
            dist2 = _softclip(dist2_unclipped)
        # FIXME: track clipped / unclipped losses
        losses["pos_clipped"] = (dist2_clipped[-1] * diffusion_weight).sum() / 3
        losses["pos_unclipped"] = (dist2_unclipped[-1] * diffusion_weight).sum() / 3
        dist2 = jnp.where(mask[None], dist2, 0)
        trajectory_pos_loss = (dist2 * diffusion_weight[None, ...]).sum(axis=1) / 3
        losses["pos"] = trajectory_pos_loss[-1]
        losses["pos_trajectory"] = trajectory_pos_loss
        total += c.trajectory_weight * trajectory_pos_loss.mean(axis=0) + c.pos_weight * trajectory_pos_loss[-1]
        # diffusion (all-atom, atom14)
        # FIXME: something here is doing weird stuff - x_loss is too small
        if not c.decouple_ae:
            atom_pos_pred = result["denoised_decoder"]["atom_pos"]
            atom_pos_gt = data["atom_pos"]
            dist2 = ((atom_pos_gt - result["denoised_decoder"]["atom_pos"]) ** 2).sum(axis=-1)
            dist2_clipped = jnp.clip(dist2, 0, clip_val ** 2)
            dist2_unclipped = dist2
            dist2 = clipped_weight * dist2_clipped + unclipped_weight * dist2_unclipped
            if c.softclip:
                dist2 = _softclip(dist2_unclipped)
            atom_mask = data["atom_mask"]
            # only apply the loss to late diffusion steps (not used in the manuscript)
            atom_mask *= late_mask[..., None]
            x_loss = (jnp.where(atom_mask, dist2, 0)).sum(axis=-1) / 3
            x_loss /= jnp.maximum(atom_mask.sum(axis=-1), 1)
            weight = base_weight
            if c.x_loss_t_weight:
                weight *= loss_weight
            x_loss = (x_loss * weight).sum()
            losses["x"] = x_loss
            total += c.x_weight * x_loss
        # rotation loss
        gt_frames, _ = extract_aa_frames(Vec3Array.from_array(data["atom_pos"]))
        if not c.decouple_ae:
            frames, _ = extract_aa_frames(Vec3Array.from_array(result["denoised_decoder"]["atom_pos"]))
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

        # FAPE ** 2
        # FIXME: how much runtime is coming from pairwise loss?
        pair_mask = batch[:, None] == batch[None, :]
        pair_mask *= mask[:, None] * mask[None, :]
        pos_gt = data["pos"]
        if c.stop_encoder_loss:
            pos_gt = jax.lax.stop_gradient(pos_gt)
        pos_gt = jnp.where(mask[:, None, None], pos_gt, 0)
        pos_gt = Vec3Array.from_array(pos_gt)
        frames_gt, _ = extract_aa_frames(jax.lax.stop_gradient(pos_gt))
        # CB distance
        distance = (pos_gt[:, None, 4] - pos_gt[None, :, 4]).norm()
        distance = jnp.where(pair_mask, distance, jnp.inf)
        # get random neighbours to compute sparse FAPE on
        mask_neighbours = mask[:, None] * mask[None, :]
        pos_gt_local = frames_gt[:, None, None].apply_inverse_to_point(pos_gt[None, :, :])
        pos_pred = Vec3Array.from_array(result["trajectory"][-1])
        frames, _ = extract_aa_frames(pos_pred)
        pos_local = frames[:, None, None].apply_inverse_to_point(pos_pred[None, :, :])
        aligned_error = (pos_local - pos_gt_local).norm2()
        fape_clipped = (jnp.clip(aligned_error, 0.0, clip_val ** 2)).mean(axis=-1)
        fape_unclipped = aligned_error.mean(axis=-1)
        if c.fape_clipped:
            fape = fape_clipped
        else:
            fape = clipped_weight * fape_clipped + unclipped_weight * fape_unclipped
        if c.softclip:
            fape = _softclip(fape_unclipped)
        fape = jnp.where(mask_neighbours, fape, 0)
        fape = fape.sum(axis=-1) / jnp.maximum(mask_neighbours.sum(axis=1), 1)
        fape = (fape * base_weight).sum(axis=-1)
        losses["fape"] = fape / 3
        total += c.fape_weight * fape / 3

        # distogram loss
        ca = pos_gt[:, 1]
        distance_gt = (ca[:, None] - ca[None, :]).norm()
        distogram_gt = jax.lax.stop_gradient(distance_one_hot(distance_gt))
        distogram_nll = -(result["distogram"] * distogram_gt).sum(axis=-1)
        distogram_nll = jnp.where(
            pair_mask, distogram_nll, 0).sum(
            axis=-1
        ) / jnp.maximum(pair_mask.sum(axis=-1), 1)
        distogram_nll = (distogram_nll * base_weight).sum()
        losses["distogram"] = distogram_nll
        total += 0.1 * distogram_nll

        # violation loss
        if not c.decouple_ae:
            res_mask = data["mask"] * late_mask
            pred_mask = get_atom14_mask(data["aa_gt"]) * res_mask[:, None]
            violation, _ = violation_loss(
                data["aa_gt"],
                data["residue_index"],
                result["denoised_decoder"]["atom_pos"],
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

class HybridDiffusionBlock(hk.Module):
    def __init__(self, config, name = "hybrid_diffn_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, pair, condition,
                 resi, chain, batch, mask, time=None,
                 pair_bias=None):
        c = self.config
        neighbours = extract_neighbours()(pos, resi, chain, batch, mask)
        if c.non_equivariant:
            pair_features, _ = non_equivarient_hybrid_pair_features(c)(local, pos, pair, neighbours, mask)
        else:
            pair_features, _ = hybrid_pair_features(c)(local, pos, pair, neighbours, mask)
        pair_features = dropout(c, pair_features)
        if c.non_equivariant:
            local_pos = Vec3Array.from_array(pos - pos[:, 1:2])
        else:
            _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        # FIXME: RMSnorm?
        local_features = local_pos.to_array().reshape(local.shape[0], -1)
        local_features /= jnp.sqrt(1e-6 + (local_features ** 2).mean(axis=-1, keepdims=True))
        local += Linear(
            local.shape[-1],
            initializer="zeros")(local_features)
        local += dropout(c, HybridPointAttention(equivariant=not c.non_equivariant)(
            CondNorm()(local, condition), 0.1 * pos,
            pair_features, neighbours,
            resi, chain, batch, mask,
            pair_bias=pair_bias))
        local += dropout(c, GatedMLP(
            local.shape[-1] * 2,
            local.shape[-1])(CondNorm()(local, condition)))
        local_norm = CondNorm()(local, condition)
        if c.learned_scale:
            out_scale = c.sigma_data * jax.nn.sigmoid(
                Linear(1, initializer="linear")(local_norm))
        else:
            out_scale = preconditioning_scale_factors(
                c, time[:, None], c.sigma_data)["out"]
        # jax.debug.print("Step scale {scale:.2f}", scale=out_scale.mean())
        pos = update_positions(pos, local_norm, scale=out_scale, non_equivariant=c.non_equivariant)
        return local, pos

class DiffusionStack(hk.Module):
    """VE Diffusion stack with input and output scaling."""
    def __init__(self, config, block, name: str | None = "diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, pair, condition,
                 time, resi, chain, batch, mask,
                 pair_bias=None):
        c = self.config
        scale = preconditioning_scale_factors(c, time, c.sigma_data)
        # Ensure that positions are centered prior to scaling
        # failing to do so will cause issues during model training
        if isinstance(scale["skip"], float):
            pos = pos * scale["skip"]
        else:
            pos = pos * scale["skip"][:, None, None]

        def stack_inner(block):
            def _inner(data):
                local, pos = data
                # run the diffusion block
                result = block(c)(
                    local, pos, pair, condition,
                    resi, chain, batch, mask,
                    time=time, pair_bias=pair_bias)
                local, pos = result
                trajectory_output = pos
                # return local & positions for the next block
                # and positions to construct a trajectory
                return (local, pos), trajectory_output

            return _inner

        diffusion_block = self.block
        if c.repeat:
            base_block = diffusion_block(c)
            diffusion_block = lambda _: base_block
        # stack = block_stack(c.diffusion_depth, c.block_size, with_state=True)(
        #     hk.remat(stack_inner(diffusion_block))
        # )
        # (local, pos), trajectory = stack((local, pos))
        # FIXME: is the loop an issue?
        # previous testing without the loop did not exhibit long runtimes
        trajectory = []
        for _ in range(c.diffusion_depth):
            (local, pos), traj = stack_inner(diffusion_block)((local, pos))
            trajectory.append(traj)
        trajectory = jnp.stack(trajectory, axis=0)

        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        return local, pos, trajectory

class StagedDiffusionStack(hk.Module):
    def __init__(self, config, embed_block, refine_block, name = "staged_diffusion_stack"):
        super().__init__(name)
        self.config = config
        self.embed_block = embed_block
        self.refine_block = refine_block

    def __call__(self, local, pos, pair, condition,
                 time, resi, chain, batch, mask,
                 pair_bias=None):
        c = self.config
        local = CondNorm()(local, condition)
        def embed_inner(block):
            def _inner(data):
                local = data
                # run the diffusion block
                result = block(c)(
                    local, pos, pair, condition,
                    resi, chain, batch, mask,
                    time=time, pair_bias=pair_bias)
                local, _ = result
                return local
            return _inner
        def refine_inner(block):
            def _inner(data):
                local, pos = data
                # run the diffusion block
                result = block(c)(
                    local, pos, pair, condition,
                    resi, chain, batch, mask,
                    time=time, pair_bias=pair_bias)
                local, pos = result
                trajectory_output = pos
                # return local & positions for the next block
                # and positions to construct a trajectory
                return (local, pos), trajectory_output
            return _inner
        embed_stack = block_stack(c.embed_depth, c.block_size, with_state=False)(
            hk.remat(embed_inner(self.embed_block)))
        refine_stack = block_stack(c.refine_depth, c.block_size, with_state=True)(
            hk.remat(refine_inner(self.refine_block)))
        local = embed_stack(local)
        local = CondNorm()(local, condition)
        # TODO: update positions
        scale = preconditioning_scale_factors(c, time, c.sigma_data)
        out_scale = c.sigma_data * jax.nn.sigmoid(
                Linear(1, initializer="linear")(local))
        start_pos = update_positions(pos * scale["skip"], local, scale=out_scale)
        (local, pos), trajectory = refine_stack((local, start_pos))
        if c.block_size > 1:
            trajectory = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        trajectory = jnp.concatenate((start_pos[None], trajectory), axis=0)

        return local, pos, trajectory

class CondNorm(hk.Module):
    def __init__(self, name = "cond_norm"):
        super().__init__(name)

    def __call__(self, data, condition):
        data_norm = hk.LayerNorm([-1], False, False)(data)
        scale = jax.nn.sigmoid(Linear(data.shape[-1], initializer="linear")(condition))
        bias = Linear(data.shape[-1], initializer="zeros")(condition)
        return scale * data_norm + bias

def hybrid_pair_features(c):
    def inner(local, pos, pair, neighbours, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = pair[axis_index(pair, 0)[:, None], neighbours]
        # FIXME
        # compute pair features based on current position
        pos = Vec3Array.from_array(pos)
        # add local features
        pair += Linear(c.pair_size, bias=False, initializer="linear")(local)[:, None]
        pair += Linear(c.pair_size, bias=False, initializer="linear")(local)[neighbours]
        # add geometric features
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        # apply 2-layer MLP
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(
            pair.shape[-1] * 2, pair.shape[-1],
            activation=jax.nn.gelu,
            final_init="linear")(pair)
        return pair, pair_mask

    return inner

def non_equivarient_hybrid_pair_features(c):
    def inner(local, pos, pair, neighbours, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = pair[axis_index(pair, 0)[:, None], neighbours]
        # compute pair features based on current position
        pos = Vec3Array.from_array(pos)
        # add local features
        pair += Linear(c.pair_size, bias=False, initializer="linear")(local)[:, None]
        pair += Linear(c.pair_size, bias=False, initializer="linear")(local)[neighbours]
        # add geometric features
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        offsets: Vec3Array = pos[:, None, :5, None] - pos[neighbours, None, :]
        directions = offsets.normalized()
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            directions.to_array().reshape(*neighbours.shape, -1))
        self_directions: Vec3Array = (pos - pos[:, 1:2]).normalized()
        pair_dot: jnp.ndarray = self_directions[:, None, :5, None].dot(self_directions[neighbours, None, :])
        pair_dot = pair_dot.reshape(*neighbours.shape, -1)
        pair_cross: Vec3Array = self_directions[:, None, :5, None].cross(self_directions[neighbours, None, :])
        pair_cross = pair_cross.to_array().reshape(*neighbours.shape, -1)
        pair_orientation = jnp.concatenate((pair_dot, pair_cross), axis=-1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_orientation)
        pos = pos.to_array()
        pair_vector = jnp.concatenate((
            0.1 * jnp.broadcast_to((pos - pos[:, 1:2])[:, None], list(neighbours.shape) + [20, 3]),
            0.1 * (pos[neighbours] - pos[:, None, 1:2])
        ), axis=-1).reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector)
        # apply 2-layer MLP
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(
            pair.shape[-1] * 2, pair.shape[-1],
            activation=jax.nn.gelu,
            final_init="linear")(pair)
        return pair, pair_mask
    return inner
