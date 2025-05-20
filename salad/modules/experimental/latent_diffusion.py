from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk

# alphafold dependencies
from salad.aflib.model.geometry import Vec3Array, Rot3Array
from salad.aflib.model.all_atom_multimer import get_atom14_mask
from salad.modules.utils.alphafold_loss import violation_loss

# basic module imports
from salad.modules.basic import (
    Linear, MLP, init_relu,
    init_zeros, init_linear,
    block_stack
)

# import geometry utils
from salad.modules.utils.geometry import (
    index_mean, index_sum, extract_aa_frames,
    extract_neighbours, distance_rbf,
    unique_chain, positions_to_ncacocb, index_align,
    single_protein_sidechains, compute_pseudo_cb,
    get_random_neighbours, get_spatial_neighbours,
    get_neighbours, axis_index, distance_one_hot)
from salad.modules.utils.dssp import assign_dssp

# sparse geometric module imports
from salad.modules.geometric import (
    SparseStructureAttention, SparseAttention, SparseSemiEquivariantPointAttention,
    sequence_relative_position,
    distance_features, direction_features, pair_vector_features,
    position_rotation_features,
    vector_mean_norm, VectorLinear
)

# TODO: finish implementing this
class LatentDiffusion(hk.Module):
    def __init__(self, config,
                 name: Optional[str] = "latent_diffusion"):
        super().__init__(name)
        self.config = config

    def prepare_condition(self, data):
        c = self.config
        aa = data["aa_gt"]
        pos = data["atom_pos"]
        pos_mask = data["atom_mask"]
        batch = data["batch_index"]
        chain = data["chain_index"]
        dssp, _, _ = assign_dssp(pos, data["batch_index"], pos_mask.any(axis=-1))
        aa_mask, dssp_mask = self.get_masks(
            chain, batch)

        # set up pair condition
        percentage = jax.random.uniform(hk.next_rng_key(), (aa.shape[0],), minval=0.2, maxval=0.8)[batch]
        pos_mask = jax.random.bernoulli(hk.next_rng_key(), percentage)
        pos_mask *= jax.random.bernoulli(hk.next_rng_key(), p=0.5, shape=batch.shape)[chain]

        # include conditioning stochastically during training
        use_condition = jax.random.bernoulli(hk.next_rng_key(), 0.5, (aa.shape[0],))
        use_condition = use_condition[data["batch_index"]]
        result = dict(condition=dict(dssp=jnp.where(dssp_mask * use_condition, dssp, 3),
                                     aa=jnp.where(aa_mask * use_condition, aa, 20),
                                     pos=pos,
                                     pos_mask=pos_mask * use_condition))
        return result
    
    def get_masks(self, chain_index, batch_index):
        def mask_me(batch, p=0.5, p_min=0.2, p_max=1.0):
            p_mask = jax.random.uniform(hk.next_rng_key(), shape=batch.shape, minval=p_min, maxval=p_max)[batch]
            bare_mask = jax.random.bernoulli(hk.next_rng_key(), p_mask)
            keep_mask = jax.random.bernoulli(hk.next_rng_key(), p=p, shape=batch.shape)[batch]
            return bare_mask * keep_mask
        sequence_mask = mask_me(batch_index, p=0.5)
        sse_mask = mask_me(batch_index, p=0.5)
        return sequence_mask, sse_mask

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

        # assign dssp
        dssp, _, _ = assign_dssp(atom_pos, batch, mask)

        # set initial backbone positions
        pos = jax.random.normal(hk.next_rng_key(), pos_ncacocb.shape)
        return dict(pos=pos, pos_gt=pos_ncacocb, pos_input=pos_ncacocb,
                    dssp=dssp, dmap=dmap, dmap_mask=dmap_mask,
                    chain_index=chain, mask=mask,
                    atom_pos=atom_pos, atom_mask=atom_mask,
                    all_atom_positions=pos_14,
                    all_atom_mask=atom_mask_14)

    def __call__(self, data):
        c = self.config
        encoder = Encoder(c)
        decoder = Decoder(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # prepare conditioning information
        data.update(self.prepare_condition(data))
        # NOTE: constraining latents is necessary for diffusion to work
        # with a trainable encoder. Otherwise, the model learns to cheat.
        # we do this by applying a parameter-less LayerNorm to the latent
        # vectors. This fixes the variance of the latent vectors to 1 and
        # bounds the achievable signal-to-noise ratio during the diffusion
        # process. Thus, the model has to actively learn to denoise.
        latent = encoder(data)
        if c.quantize:
            projection = hk.get_parameter(
                "qproj", (latent.shape[-1], 12), dtype=jnp.float32, init=init_linear())
            qdata = jnp.einsum("ij,jk->ik", latent, projection)
            rounded = jax.lax.stop_gradient(qdata)
            rounded = jnp.where(rounded > 0, 1, -1)
            qdata = rounded + qdata - jax.lax.stop_gradient(qdata)
            latent = jnp.einsum("ik,jk->ij", qdata, projection)
            latent = hk.LayerNorm([-1], False, False)(latent)
        else:
            latent = hk.LayerNorm([-1], False, False)(latent)
        data["clean_latent"] = latent
        latent, time = self.prepare_latent_diffusion(latent, data)
        data["time"] = time
        data["latent"] = latent

        # self-conditioning
        def iteration_body(i, prev):
            result = decoder(data, prev)
            prev = dict(
                pos=result["pos"],
                local=result["local"],
                latent=result["latent"]
            )
            return jax.lax.stop_gradient(prev)
        prev = dict(
            pos=data["pos"],
            local=jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32),
            latent=jnp.zeros((data["pos"].shape[0], c.local_size))
        )
        if not hk.running_init():
            if c.eval:
                count = 3
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

    def prepare_latent_diffusion(self, latent, data):
        c = self.config
        batch = data["batch_index"]
        noise = jax.random.normal(hk.next_rng_key(), latent.shape)
        time = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
        if "time" in data:
            time = data["time"] * jnp.ones_like(time)
        s = 0.01
        time = jnp.cos((time + s) / (1 + s) * jnp.pi / 2) / jnp.cos(s / (1 + s) * jnp.pi / 2)
        time = jnp.sqrt(jnp.clip(1 - time ** 2, 0, 1))
        latent = jnp.sqrt(1 - time[:, None] ** 2) * latent + time[:, None] * noise * 2.0
        return latent, time


# class AugmentPos(hk.Module):
#     def __init__(self, config, name: Optional[str] = "augment_pos"):
#         super().__init__(name)
#         self.config = config

#     def prepare_features(self, positions, resi, chain, batch, mask):
#         c = self.config
#         positions = Vec3Array.from_array(positions)
#         frames, local_positions = extract_aa_frames(positions)
#         augment = VectorLinear(
#             11,
#             initializer=init_linear())(local_positions)
#         augment = vector_mean_norm(augment)
#         local_positions = jnp.concatenate((
#             local_positions.to_array()[:, :5],
#             augment.to_array()
#         ), axis=-2)
#         local_positions = Vec3Array.from_array(local_positions)
#         positions = frames[:, None].apply_to_point(local_positions)
#         neighbours = get_spatial_neighbours(c.num_neighbours)(
#             positions[:, 4], batch, mask)
#         _, local_positions = extract_aa_frames(positions)
#         dist = local_positions.norm()

#         local_features = [
#             local_positions.normalized().to_array().reshape(
#                 local_positions.shape[0], -1),
#             distance_rbf(dist, 0.0, 22.0, 16).reshape(
#                 local_positions.shape[0], -1),
#             jnp.log(dist + 1)
#         ]
#         local = MLP(c.local_size * 4, c.local_size, activation=jax.nn.gelu,
#                     bias=False)(
#             jnp.concatenate(local_features, axis=-1))
        
#         positions = positions.to_array()
#         local = hk.LayerNorm([-1], True, True)(local)
#         return local, positions, neighbours, resi, chain, batch, mask

#     def __call__(self, pos, resi, chain, batch, mask):
#         c = self.config
#         local, pos, neighbours = self.prepare_features(
#             pos, resi, chain, batch, mask
#         )
#         local = EncoderStack(c, 2)(
#             local, pos, neighbours, resi, chain, batch, mask)
#         # generate augmented vector features from encoder representation
#         frames, local_positions = extract_aa_frames(
#             Vec3Array.from_array(pos))
#         augment = local_positions[:, 5:]
#         update = MLP(2 * local.shape[-1], augment.shape[1] * 3,
#             activation=jax.nn.gelu, final_init=init_linear())(local)
#         update = update.reshape(update.shape[0], augment.shape[1], 3)
#         update = Vec3Array.from_array(update)
#         augment += update
#         local_positions = local_positions[:, :5]
#         augment = vector_mean_norm(augment)
#         # combine augmented features with backbone / atom positions
#         local_positions = jnp.concatenate((
#             local_positions.to_array(), augment.to_array()), axis=-2)
#         pos = frames[:, None].apply_to_point(
#             Vec3Array.from_array(local_positions)).to_array()
#         return local, pos

# TODO
class LatentDiffusionPredict(LatentDiffusion):
    def __call__(self, data):
        c = self.config
        encoder = Encoder(c)
        decoder = Decoder(c)

        # prepare conditioning information
        data.update(self.prepare_condition(data))
        # TODO
        latent, time = self.prepare_latent_diffusion(latent, data)
        data["time"] = time
        data["latent"] = latent

        # self-conditioning
        def iteration_body(i, prev):
            result = decoder(data, prev)
            prev = dict(
                pos=result["pos"],
                local=result["local"],
                latent=result["latent"]
            )
            return jax.lax.stop_gradient(prev)
        prev = dict(
            pos=data["pos"],
            local=jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32),
            latent=jnp.zeros((data["pos"].shape[0], c.latent_size))
        )
        count = 3
        prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = decoder(data, prev)
        result["aatype"] = jnp.argmax(result["aa"], axis=-1)
        prev = dict(
            pos=result["pos"],
            local=result["local"],
            latent=result["latent"]
        )
        return result, prev

class LatentDiffusionInference(LatentDiffusion):
    def __call__(self, data):
        c = self.config
        encoder = Encoder(c)
        decoder = Decoder(c)

        pos = jax.random.normal(hk.next_rng_key(), (data["aa_gt"].shape[0], 5, 3))
        data["pos"] = pos
        # prepare conditioning information
        data.update(self.prepare_condition(data))
        latent = data["latent"]
        # latent = hk.LayerNorm([-1], False, False)(latent)
        latent, time = self.prepare_latent_diffusion(latent, data)
        data["time"] = time
        data["latent"] = latent

        # self-conditioning
        def iteration_body(i, prev):
            result = decoder(data, prev)
            prev = dict(
                pos=result["pos"],
                local=result["local"],
                latent=result["predicted_latent"]
            )
            return jax.lax.stop_gradient(prev)
        prev = dict(
            pos=data["pos"],
            local=jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32),
            latent=jnp.zeros((data["pos"].shape[0], c.local_size))
        )
        count = c.num_recycle
        prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = decoder(data, prev)
        result["aatype"] = jnp.argmax(result["aa"], axis=-1)
        prev = dict(
            pos=result["pos"],
            local=result["local"],
            latent=result["predicted_latent"]
        )
        return result, prev

    def prepare_condition(self, data):
        c = self.config
        aa = data["aa_gt"]
        # TODO: currently condition is deactivated
        # pos = data["atom_pos"]
        # pos_mask = data["atom_mask"]
        # batch = data["batch_index"]
        # chain = data["chain_index"]
        # dssp, _, _ = assign_dssp(pos, data["batch_index"], pos_mask.any(axis=-1))
        # aa_mask, dssp_mask = self.get_masks(
        #     chain, batch)

        # # set up pair condition
        # percentage = jax.random.uniform(hk.next_rng_key(), (aa.shape[0],), minval=0.2, maxval=0.8)[batch]
        # pos_mask = jax.random.bernoulli(hk.next_rng_key(), percentage)
        # pos_mask *= jax.random.bernoulli(hk.next_rng_key(), p=0.5, shape=batch.shape)[chain]

        # # include conditioning stochastically during training
        # use_condition = jax.random.bernoulli(hk.next_rng_key(), 0.5, (aa.shape[0],))
        # use_condition = use_condition[data["batch_index"]]
        result = dict(condition=dict(dssp=3 * jnp.ones_like(aa),
                                     aa=20 * jnp.ones_like(aa),
                                     pos=jnp.zeros((aa.shape[0], 5, 3), dtype=jnp.float32),
                                     pos_mask=jnp.zeros((aa.shape[0],), dtype=jnp.bool_)))
        return result

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

class EncoderBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "encoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos,
                 resi, chain, batch, mask):
        c = self.config
        # embed pair features
        neighbours = extract_neighbours(16, 16, 32)(
            Vec3Array.from_array(pos),
            resi, chain, batch, mask)
        pair_feature_function = aa_decoder_pair_features
        if c.no_resi_encoder:
            pair_feature_function = no_resi_pair_features
        pair, pair_mask = pair_feature_function(c)(
            Vec3Array.from_array(pos), neighbours, resi, chain, batch, mask)
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
                hk.LayerNorm([-1], True, True)(features),
                pos, chain, batch, mask)
        return features

class EncoderStack(hk.Module):
    def __init__(self, config, depth=None, name: Optional[str] = "encoder_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 3

    def __call__(self, local, pos,
                 resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                data = block(c)(data, pos,
                                resi, chain,
                                batch, mask)
                return data
            return _inner
        stack = block_stack(
            self.depth, block_size=1, with_state=False)(
                hk.remat(stack_inner(EncoderBlock)))
        local = hk.LayerNorm([-1], True, True)(stack(local))
        return local

class Encoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "encoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        local, pos, resi, chain, batch, mask = self.prepare_features(data)
        local = EncoderStack(c, depth=c.encoder_depth)(
            local, pos, resi, chain, batch, mask)
        if c.noembed:
            return local
        local = hk.LayerNorm([-1], True, True)(local)
        return Linear(c.latent_size, initializer=init_linear(), bias=False)(local)
    
    def prepare_features(self, data):
        c = self.config
        pos = data["pos_input"]
        # FIXME: This could be the cause of the model learning to encode
        # resi implicitly. As another way to circumvent this, remove
        # any mention of \Delta resi in the encoder
        # if c.noise_encoder and not c.eval:
        #     pos += c.noise_encoder * jax.random.normal(hk.next_rng_key(), shape=pos.shape)
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        pos = Vec3Array.from_array(pos)
        neighbours = extract_neighbours(5, 5, 0)(
            pos, resi, chain, batch, mask)
        local_features = init_local_features(c)(
            pos, neighbours, resi, chain, batch, mask)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
        local = hk.LayerNorm([-1], True, True)(local)

        return local, pos.to_array(), resi, chain, batch, mask

class ConditionEncoder(hk.Module):
    def __init__(self, config, name: Optional[str] = None):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        pos = data["condition"]["pos"]
        pos = positions_to_ncacocb(pos)
        pos_mask = data["condition"]["pos_mask"]
        ce_stack = EncoderStack(c, 2, name="condition_encoder_stack")
        dssp = Linear(c.local_size, bias=False)(
            jax.nn.one_hot(data["condition"]["dssp"], 3, axis=-1))
        aa = Linear(c.local_size, bias=False)(
            jax.nn.one_hot(data["condition"]["aa"], 20, axis=-1))
        condition_local = dssp + aa
        # dist = Vec3Array.from_array(pos[:, :, None] - pos[:, None, :]).norm()
        # dist = dist.reshape(dist.shape[0], -1)
        # dist = Linear(c.local_size, bias=False)(
        #     distance_rbf(dist, 0.0, 5.0, 16).reshape(dist.shape[0], -1))
        # dist = jnp.where(pos_mask[:, None], dist, 0)
        # condition_local += dist
        # condition_local += jnp.where(
        #     pos_mask[..., None], ce_stack(
        #         condition_local, pos,
        #         resi, chain, batch, pos_mask), 0)
        return condition_local

class Decoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config
        self.decoder_block = DecoderBlock

    def __call__(self, data, prev):
        c = self.config
        decoder_module = DecoderStack
        decoder_stack = decoder_module(c, self.decoder_block)
        aa_decoder = AADecoder(c)
        local, latent, pos, resi, chain, batch, mask = self.prepare_features(data, prev)
        sup_neighbours = get_random_neighbours(c.fape_neighbours)(
            Vec3Array.from_array(data["pos_gt"][:, -1]), batch, mask)
        local, latent, pos, trajectory = decoder_stack(
            local, latent, pos, resi, chain, batch, mask)
        pair, pair_mask = distogram_pair_features(c)(
            local, pos, sup_neighbours, resi, chain, batch, mask)
        distogram = jax.nn.log_softmax(
            Linear(64, bias=False, initializer="zeros")(pair))
        # predict & handle sequence, losses etc.
        result = dict()
        result["latent"] = latent
        result["trajectory"] = trajectory
        result["sup_neighbours"] = sup_neighbours
        result["distogram"] = distogram
        result["local"] = local
        result["pos"] = pos
        result["predicted_latent"] = latent
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
            local = jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32),
            latent = jnp.zeros((data["pos"].shape[0], c.latent_size), dtype=jnp.float32)
        )

    def prepare_features(self, data, prev):
        c = self.config
        pos = prev["pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        latent = data["latent"]
        mask = data["mask"]
        prev_latent = prev["latent"]
        condition = ConditionEncoder(c)(data)
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1),
            latent,
            hk.LayerNorm([-1], True, True)(prev_latent)
        ]
        # if c.time_embedding and "time" in data:
        #     time = data["time"]
        #     time = distance_rbf(time, 0, 1.0, bins=200)
        #     local_features.append(time)
        local_features = jnp.concatenate(
            local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
        local = hk.LayerNorm([-1], True, True)(local)
        local += hk.LayerNorm([-1], True, True)(prev["local"])
        local += hk.LayerNorm([-1], True, True)(condition)

        return local, latent, pos, resi, chain, batch, mask

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
        fape_base = (traj_local - pos_gt_local).norm2()
        fape_clipped = jnp.clip(fape_base, 0.0, c.clip_fape)
        if c.unclipped_weight:
            fape_clipped = fape_base * c.unclipped_weight + fape_clipped
        if c.no_fape2:
            # use FAPE instead of FAPE ** 2
            fape_clipped = jnp.sqrt(jnp.maximum(fape_clipped, 1e-6))
        fape_traj = fape_clipped.mean(axis=-1)
        fape_traj = jnp.where(mask_neighbours[None], fape_traj, 0)
        fape_traj = fape_traj.sum(axis=-1) / jnp.maximum(mask_neighbours.sum(axis=1)[None], 1)
        fape_traj = (fape_traj * base_weight).sum(axis=-1)
        losses["fape"] = fape_traj[-1] / 3
        losses["fape_trajectory"] = fape_traj.mean() / 3
        fape_loss = (c.fape_weight * fape_traj[-1] + c.fape_trajectory_weight * fape_traj.mean()) / 3
        total += fape_loss

        # Kabsch RMSD loss
        if c.kabsch_rmsd:
            last = traj[-1]
            pos_gt = jax.lax.stop_gradient(index_align(pos_gt, last, batch, mask))
            pos_loss_unclipped = (pos_gt[None] - traj).norm2()
            pos_loss_clipped = jnp.clip(pos_loss_unclipped, 0, 100.0)
            pos_loss = pos_loss_clipped
            if c.unclipped_weight:
                pos_loss = pos_loss_unclipped * c.unclipped_weight + pos_loss
            pos_loss = pos_loss.mean(axis=-1)
            pos_loss *= base_weight[None]
            pos_loss = pos_loss.sum(axis=-1)
            losses["kabsch_rmsd"] = pos_loss[-1] / 3
            losses["kabsch_rmsd_trajectory"] = pos_loss.mean() / 3
            pos_loss = (c.fape_weight * pos_loss[-1] + c.fape_trajectory_weight * pos_loss.mean()) / 3
            total += pos_loss

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

        # distogram loss
        cb_gt = pos_gt[:, -1]
        sup_neighbours = result["sup_neighbours"]
        sup_mask = (sup_neighbours != -1) * mask[:, None] * mask[sup_neighbours]
        dist_gt = (cb_gt[:, None] - cb_gt[sup_neighbours]).norm()
        dist_one_hot = distance_one_hot(dist_gt, 0, 22.0, 64)
        distogram_nll = -(result["distogram"] * dist_one_hot).sum(axis=-1)
        distogram_nll = jnp.where(sup_mask, distogram_nll, 0).sum(axis=-1)
        distogram_nll /= jnp.maximum(sup_mask.sum(axis=1), 1)
        distogram_nll = (distogram_nll * base_weight).sum()
        losses["distogram"] = distogram_nll
        total += 2.0 * distogram_nll
        # additional denoising losses
        # unweighted loss
        raw_loss = ((data["clean_latent"] - result["predicted_latent"]) ** 2).mean(axis=-1)
        weighted_loss = (jnp.where(mask, raw_loss, 0) * base_weight).sum()
        losses["latent"] = weighted_loss
        total += 10.0 * weighted_loss
        if c.violation_scale:
            res_mask = data["mask"]
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
            total += c.violation_scale * violation.mean()

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

    def __call__(self, local, latent, pos,
                 resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, latent, pos = data
                # run the decoder block
                features, latent, pos = block(c)(
                    features, latent, pos,
                    resi, chain, batch, mask,
                    )
                trajectory_output = pos
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, latent, pos), trajectory_output
            return _inner
        decoder_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(decoder_block)))
        (local, latent, pos), trajectory = stack((local, latent, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
        return local, latent, pos, trajectory

def structure_augmentation_params(pos, batch, mask):
    # center positions
    center = index_mean(pos[:, 1], batch, mask[:, None])
    # centering + random translation
    translation = jax.random.normal(hk.next_rng_key(), pos[:, 1].shape)[batch]
    # random rotation
    rotation = random_rotation(batch)
    return center, translation, rotation

def apply_structure_augmentation(pos, center, translation, rotation):
    # center positions
    pos -= center[:, None]
    # apply transformation
    pos = rotation[:, None].apply_to_point(Vec3Array.from_array(pos)).to_array()
    pos += translation[:, None]
    return pos

def apply_inverse_structure_augmentation(pos, center, translation, rotation):
    # invert translation
    pos -= translation[:, None]
    # invert rotation
    pos = rotation[:, None].apply_inverse_to_point(Vec3Array.from_array(pos)).to_array()
    # move to center
    pos += center[:, None]
    return pos

def structure_augmentation(pos, batch, mask):
    # get random augmentation parameters
    center, translation, rotation = structure_augmentation_params(pos, batch, mask)
    # apply random augmentation
    pos = apply_structure_augmentation(pos, center, translation, rotation)
    return pos

def random_rotation(batch):
    x = jax.random.normal(hk.next_rng_key(), (batch.shape[0], 3))[batch]
    y = jax.random.normal(hk.next_rng_key(), (batch.shape[0], 3))[batch]
    result = Rot3Array.from_two_vectors(
        Vec3Array.from_array(x),
        Vec3Array.from_array(y))
    return result

def no_resi_pair_features(c):
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
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

def aa_decoder_pair_features(c):
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, pseudo_chains=True)(
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

def init_local_features(c):
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        if c.no_resi_encoder:
            pair = 0.0
        else:
            pair = Linear(c.pair_size, bias=False, initializer="linear")(
                sequence_relative_position(8, one_hot=True, pseudo_chains=True)(
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
        pair = MLP(c.pair_size * 2, c.pair_size, activation=jax.nn.gelu)(pair)
        pair = jnp.where(pair_mask[..., None], pair, 0)
        local = pair.sum(axis=1) / jnp.maximum(pair_mask.sum(axis=-1)[..., None], 1)
        local = Linear(c.local_size, bias=False)(local)
        return local
    return inner

def decoder_pair_features(c):
    def inner(pos, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
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

def distogram_pair_features(c):
    def inner(local, pos, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        left = Linear(c.pair_size, bias=False)(local)
        right = Linear(c.pair_size, bias=False)(local)
        pair = left[:, None] + right[neighbours]
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
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

    def __call__(self, features, latent, pos,
                 resi, chain, batch, mask):
        c = self.config

        current_neighbours = extract_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=32)(
                Vec3Array.from_array(pos),
                resi, chain, batch, mask)

        pair, pair_mask = decoder_pair_features(c)(
            pos, current_neighbours,
            resi, chain, batch, mask)
        features += Linear(
            features.shape[-1], bias=False, initializer=init_zeros())(latent)
        features += SparseStructureAttention(c)(
                hk.LayerNorm([-1], True, True)(features),
                pos / c.sigma_data, pair, pair_mask,
                current_neighbours, resi, chain, batch, mask)
        # global update of local features
        features += DecoderUpdate(c)(
                hk.LayerNorm([-1], True, True)(features),
                pos, chain, batch, mask)
        local_norm = hk.LayerNorm([-1], True, True)(features)
        # update latent
        latent_update = MLP(features.shape[-1] * 2, latent.shape[-1],
                            activation=jax.nn.gelu, final_init="zeros")(local_norm)
        latent += latent_update
        # update positions
        split_scale_out = c.sigma_data
        pos = update_positions(pos, local_norm,
                               scale=split_scale_out,
                               symm=c.symm)
        return features, latent, pos.astype(local_norm.dtype)

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
