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
    position_rotation_features
)

class StructureAutoencoder(hk.Module):
    def __init__(self, config,
                 name: Optional[str] = "structure_autoencoder"):
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
        if c.codebook_size:
            vq = VQState if c.state else VQ
            mapped_axes = []
            if c.rebatch:
                mapped_axes.append("rebatch_ax")
            if c.multigpu:
                mapped_axes.append("batch_ax")
            quantize = vq(c.codebook_size, c.affine, mapped_axes=mapped_axes)
        if c.fsq:
            fsq = FSQ()
        if c.hallucination:
            error = Error(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # optionally apply noise to the inputs
        if c.input_diffusion:
            clean_latent = encoder(data)
            data["clean_latent"] = clean_latent
            data.update(self.prepare_input_diffusion(data))
        latent = encoder(data)
        # optionally apply noise to the latents
        if c.latent_diffusion:
            # NOTE: constraining latents is necessary for diffusion to work
            # with a trainable encoder. Otherwise, the model learns to cheat.
            # we do this by applying a parameter-less LayerNorm to the latent
            # vectors. This fixes the variance of the latent vectors to 1 and
            # bounds the achievable signal-to-noise ratio during the diffusion
            # process. Thus, the model has to actively learn to denoise.
            latent = hk.LayerNorm([-1], False, False)(latent)
            data["clean_latent"] = latent
            latent, time = self.prepare_latent_diffusion(latent, data)
            if not c.vp_diffusion:
                data["skip_latent"] = latent / jnp.maximum(1 + time[:, None] ** 2, 1e-3)
                latent = latent / jnp.maximum(jnp.sqrt(1 + time[:, None] ** 2), 1e-3)
            data["time"] = time
        if c.codebook_size:
            if c.state:
                latent, codebook_index, codebook_losses, state_update = quantize(latent, data["mask"])
            else:
                latent, codebook_index, codebook_losses = quantize(latent, data["mask"])
            data["codebook_index"] = codebook_index
        if c.fsq:
            latent, _ = fsq(latent)
        data["latent"] = latent
        if c.hallucination:
            latent = self.add_noise(latent)
            data["latent"] = jax.lax.stop_gradient(latent)

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
        if c.hallucination:
            result.update(error(latent, result))
        if c.codebook_size:
            result["codebook_losses"] = codebook_losses
        total, losses = decoder.loss(data, result)
        if c.hallucination:
            error_total, error_losses = error.loss(data, result)
            total += error_total
            losses.update(error_losses)
        out_dict = dict(
            results=result,
            losses=losses
        )
        if c.codebook_size and c.state:
            out_dict["_state_update"] = state_update
        return total, out_dict

    def add_noise(self, latent, batch):
        noise_level = jax.random.normal(hk.next_rng_key(), batch.shape)[batch]
        noise_level = jnp.exp(noise_level)
        noise = jax.random.normal(hk.next_rng_key(), latent.shape) * noise_level
        return latent + noise

    def prepare_input_diffusion(self, data):
        c = self.config
        batch = data["batch_index"]
        pos = data["pos_input"]
        pos -= index_mean(pos[:, 1], batch, data["mask"][:, None])[:, None] # FIXME
        time = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
        if "time" in data:
            time = data["time"] * jnp.ones_like(time)
        s = 0.01
        time = jnp.cos((time + s) / (1 + s) * jnp.pi / 2) / jnp.cos(s / (1 + s) * jnp.pi / 2)
        time = jnp.sqrt(jnp.clip(1 - time ** 2, 0, 1))
        noise = c.sigma_data * jax.random.normal(hk.next_rng_key(), pos.shape)
        pos = jnp.sqrt(1 - time[:, None, None] ** 2) * pos + time[:, None, None] * noise
        return dict(pos_input=pos, time=time)

    def prepare_latent_diffusion(self, latent, data):
        c = self.config
        batch = data["batch_index"]
        noise = jax.random.normal(hk.next_rng_key(), latent.shape)
        if c.vp_diffusion:
            time = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
        else:
            time = jnp.exp(1.0 + 1.2 * jax.random.normal(hk.next_rng_key(), batch.shape)[batch])
        # FIXME: allow time input
        if "time" in data:
            time = data["time"] * jnp.ones_like(time)
        if c.vp_diffusion:
            s = 0.01
            time = jnp.cos((time + s) / (1 + s) * jnp.pi / 2) / jnp.cos(s / (1 + s) * jnp.pi / 2)
            time = jnp.sqrt(jnp.clip(1 - time ** 2, 0, 1))
            latent = jnp.sqrt(1 - time[:, None] ** 2) * latent + time[:, None] * noise * 5.0
        else:
            latent = latent + time[:, None] * noise
        return latent, time

class StructureDecoder(StructureAutoencoder):
    def __init__(self, config,
                 name: Optional[str] = "structure_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        encoder = assign_state(c, c.param_path)
        decoder = Decoder(c)

        data.update(self.prepare_data(data))
        latent, codebook_index = encoder(hk.next_rng_key(), data)
        latent = jax.lax.stop_gradient(latent)
        data["latent"] = latent
        if c.hallucination:
            latent = self.add_noise(latent)
            data["latent"] = jax.lax.stop_gradient(latent)

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
        if c.hallucination:
            result.update(error(latent, result))
        total, losses = decoder.loss(data, result)
        if c.hallucination:
            error_total, error_losses = error.loss(data, result)
            total += error_total
            losses.update(error_losses)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

    def add_noise(self, latent, batch):
        noise_level = jax.random.normal(hk.next_rng_key(), batch.shape)[batch]
        noise_level = jnp.exp(noise_level)
        noise = jax.random.normal(hk.next_rng_key(), latent.shape) * noise_level
        return latent + noise

class StructureAutoencoderInference(StructureAutoencoder):
    def __call__(self, data):
        c = self.config
        encoder = Encoder(c)
        decoder = Decoder(c)
        if c.codebook_size:
            vq = VQState if c.state else VQ
            quantize = vq(c.codebook_size)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # optionally apply noise to the inputs
        if c.input_diffusion:
            clean_latent = encoder(data)
            data["clean_latent"] = clean_latent
            data.update(self.prepare_input_diffusion(data))
        latent = encoder(data)
        # optionally apply noise to the latents
        if c.latent_diffusion:
            if "latent" in data:
                latent = data["latent"]
            # FIXME
            # latent = hk.LayerNorm([-1], False, False)(latent)
            data["clean_latent"] = latent
            latent, time = self.prepare_latent_diffusion(latent, data)
            if not c.vp_diffusion:
                data["skip_latent"] = latent / jnp.maximum(1 + time[:, None] ** 2, 1e-3)
                latent = latent / jnp.maximum(jnp.sqrt(1 + time[:, None] ** 2), 1e-3)
            data["time"] = time
            jax.debug.print("time {time}", time=data["time"][0])
        if c.codebook_size:
            latent, codebook_index, _ = quantize(latent, data["mask"])
        data["latent"] = latent

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

        mask = data["mask"]
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(mask.sum(), 1)
        perplexity = jnp.exp(aa_nll)
        aatype = jnp.argmax(result["aa"], axis=-1)
        recovery = ((data["aa_gt"] == aatype) * mask).sum() / mask.sum()
        pos_gt = data["pos_gt"]
        pos = result["pos"]
        pos_gt = index_align(pos_gt, pos, data["batch_index"], mask)
        di2 = ((pos[:, 1] - pos_gt[:, 1]) ** 2).sum(axis=-1)
        rmsd_ca = jnp.sqrt((di2 * mask).sum() / mask.sum())
        d02 = (1.24 * (mask.sum() - 15) ** (1 / 3) - 1.8) ** 2
        inner = 1 / (1 + di2 / d02) * mask
        tm = inner.sum() / mask.sum()
        dca_gt = jnp.linalg.norm(pos_gt[:, None, 1] - pos_gt[None, :, 1], axis=-1)
        dca = jnp.linalg.norm(pos[:, None, 1] - pos[None, :, 1], axis=-1)
        pair_mask = mask[:, None] * mask[None, :]
        derr = abs(dca_gt - dca) * pair_mask
        threshold = jnp.array([0.5, 1.0, 2.0, 4.0])
        Rinc = 15.0
        pair_mask *= dca_gt < Rinc
        in_threshold = (derr[..., None] < threshold) * pair_mask[..., None]
        lddt_ca = (in_threshold.sum(axis=1) / jnp.maximum(pair_mask[..., None].sum(axis=1), 1)).mean(axis=-1)
        lddt_ca = jnp.where(mask, lddt_ca, 0)
        res_mask = data["mask"]
        pred_mask = get_atom14_mask(aatype) * res_mask[:, None]
        violation, _ = violation_loss(aatype,
                                      data["residue_index"],
                                      result["atom_pos"],
                                      pred_mask,
                                      res_mask,
                                      clash_overlap_tolerance=1.5,
                                      violation_tolerance_factor=2.0,
                                      chain_index=data["chain_index"],
                                      batch_index=data["batch_index"],
                                      per_residue=False)
        violation_error = violation.mean()

        latent = data["latent"]
        if "predicted_latent" in result:
            print("returning predicted latent")
            latent = result["predicted_latent"]
        out = dict(
            atom_pos=result["atom_pos"],
            aatype=aatype,
            latent=latent,
            local=result["local"],
            perplexity=perplexity,
            recovery=recovery,
            rmsd_ca=rmsd_ca,
            tm=tm,
            lddt=lddt_ca,
            time=data["time"],
            dssp=data["dssp"],
            violation=violation_error)
        if c.codebook_size:
            out["codebook_index"] = codebook_index
        return out
    
class StructureDecoderInference(StructureDecoder):
    def __call__(self, data):
        c = self.config
        encoder = assign_state(c, c.param_path)
        decoder = Decoder(c)

        #data.update(self.prepare_data(data))
        latent, codebook_index = encoder(hk.next_rng_key(), data)
        latent = jax.lax.stop_gradient(latent)
        data["latent"] = latent

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

        mask = data["mask"]
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(mask.sum(), 1)
        perplexity = jnp.exp(aa_nll)
        aatype = jnp.argmax(result["aa"], axis=-1)
        recovery = ((data["aa_gt"] == aatype) * mask).sum() / mask.sum()
        pos_gt = data["pos_gt"]
        pos = result["pos"]
        pos_gt = index_align(pos_gt, pos, data["batch_index"], mask)
        di2 = ((pos[:, 1] - pos_gt[:, 1]) ** 2).sum(axis=-1)
        rmsd_ca = jnp.sqrt((di2 * mask).sum() / mask.sum())
        d02 = (1.24 * (mask.sum() - 15) ** (1 / 3) - 1.8) ** 2
        inner = 1 / (1 + di2 / d02) * mask
        tm = inner.sum() / mask.sum()
        dca_gt = jnp.linalg.norm(pos_gt[:, None, 1] - pos_gt[None, :, 1], axis=-1)
        dca = jnp.linalg.norm(pos[:, None, 1] - pos[None, :, 1], axis=-1)
        pair_mask = mask[:, None] * mask[None, :]
        derr = abs(dca_gt - dca) * pair_mask
        threshold = jnp.array([0.5, 1.0, 2.0, 4.0])
        Rinc = 15.0
        pair_mask *= dca_gt < Rinc
        in_threshold = (derr[..., None] < threshold) * pair_mask[..., None]
        lddt_ca = (in_threshold.sum(axis=1) / jnp.maximum(pair_mask[..., None].sum(axis=1), 1)).mean(axis=-1)
        lddt_ca = jnp.where(mask, lddt_ca, 0)

        out = dict(
            atom_pos=result["atom_pos"],
            aatype=aatype,
            latent=data["latent"],
            local=result["local"],
            perplexity=perplexity,
            recovery=recovery,
            rmsd_ca=rmsd_ca,
            tm=tm,
            lddt=lddt_ca,
            dssp=data["dssp"])
        if c.codebook_size:
            out["codebook_index"] = codebook_index
        return out
    
class AssignState(StructureAutoencoder):
    def __call__(self, data):
        c = self.config
        encoder = Encoder(c)
        if c.codebook_size:
            vq = VQState if c.state else VQ
            quantize = vq(c.codebook_size)
        data.update(self.prepare_data(data))
        latent = encoder(data)
        if c.codebook_size:
            latent, codebook_index, _ = quantize(latent, data["mask"])
            return latent, codebook_index
        return latent, None

def assign_state(config, param_path):
    import pickle
    apply = hk.transform(lambda x: AssignState(config)(x)).apply
    with open(param_path, "rb") as f:
        params = pickle.load(f)
    def inner(key, data):
        return apply(params, key, data)
    return inner

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
        pair, pair_mask = aa_decoder_pair_features(c)(
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

# TODO TODO TODO TODO
class Error(hk.Module):
    def __init__(self, config, name: Optional[str] = "error"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        local, pos, resi, chain, batch, mask = self.prepare_features(data)
        local = EncoderStack(c, depth=c.error_depth, name="error_stack")(
            local, pos, resi, chain, batch, mask)
        local = hk.LayerNorm([-1], True, True)(local)
        InnerDistogram(c)(local, resi, chain, batch, sup_neighbours)
        return Linear(20, initializer=init_linear(), bias=False)(local)
    
    def prepare_features(self, data):
        c = self.config
        pos = data["pos_gt"]
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
        if c.noise_encoder and not c.eval:
            pos += c.noise_encoder * jax.random.normal(hk.next_rng_key(), shape=pos.shape)
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
        if c.time_embedding and c.input_diffusion and "time" in data:
            time = data["time"]
            time = distance_rbf(time, 0, 1, bins=100)
            local += Linear(
                local.shape[-1], bias=False, initializer="linear")(time)
        local = hk.LayerNorm([-1], True, True)(local)

        return local, pos.to_array(), resi, chain, batch, mask

class Decoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config
        self.decoder_block = DecoderBlock

    def __call__(self, data, prev):
        c = self.config
        decoder_module = DecoderStack
        if c.equivariance == "nonequivariant":
            decoder_module = NonEquivariantDecoderStack
            self.decoder_block = NonEquivariantDecoderBlock
        elif c.equivariance == "semiequivariant":
            decoder_module = SemiEquivariantDecoderStack
            self.decoder_block = SemiEquivariantDecoderBlock
        decoder_stack = decoder_module(c, self.decoder_block)
        aa_decoder = AADecoder(c)
        local, pos, resi, chain, batch, mask = self.prepare_features(data, prev)
        sup_neighbours = get_random_neighbours(c.fape_neighbours)(
            Vec3Array.from_array(data["pos_gt"][:, -1]), batch, mask)
        local, pos, trajectory, sup_distogram = decoder_stack(
            local, pos, resi, chain, batch, mask,
            sup_neighbours)
        # predict & handle sequence, losses etc.
        result = dict()
        result["latent"] = data["latent"]
        result["trajectory"] = trajectory
        result["sup_neighbours"] = sup_neighbours
        result["sup_distogram"] = sup_distogram
        result["local"] = local
        result["pos"] = pos
        if c.input_diffusion or c.latent_diffusion:
            latent_decoder = MLP(
                c.local_size * 2,
                c.local_size if c.noembed else c.latent_size,
                bias=False,
                final_init=init_zeros(), activation=jax.nn.gelu,
                name="latent_decoder")
            latent_update = latent_decoder(
                jnp.concatenate((
                    #local,
                    hk.LayerNorm([-1], True, True)(local),
                    data["latent"]), axis=-1))
            time = data["time"]
            if c.vp_diffusion:
                predicted_latent = data["latent"] + latent_update
            else:
                predicted_latent = data["skip_latent"] + latent_update * time[:, None] / jnp.sqrt(1 + time[:, None] ** 2)
            result["predicted_latent"] = predicted_latent
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
        latent = data["latent"]
        mask = data["mask"]
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1),
            latent
        ]
        if c.time_embedding and c.latent_diffusion and "time" in data:
            time = data["time"]
            time = distance_rbf(time, 0, 80.0, bins=200)
            local_features.append(time)
        local_features.append(hk.LayerNorm([-1], True, True)(prev["local"]))
        local_features = jnp.concatenate(
            local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
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

        # sup distogram loss
        if c.distogram_block != "none":
            cb_gt = pos_gt[:, -1]
            sup_neighbours = result["sup_neighbours"]
            sup_mask = (sup_neighbours != -1) * mask[:, None] * mask[sup_neighbours]
            dist_gt = (cb_gt[:, None] - cb_gt[sup_neighbours]).norm()
            dist_one_hot = distance_one_hot(dist_gt, 0, 22.0, 16)
            distogram_nll = -(result["sup_distogram"] * dist_one_hot[None]).sum(axis=-1)
            distogram_nll = jnp.where(sup_mask, distogram_nll, 0).sum(axis=-1)
            distogram_nll /= jnp.maximum(sup_mask.sum(axis=1), 1)
            distogram_nll = (distogram_nll * base_weight).sum(axis=1)
            losses["distogram"] = distogram_nll[-1]
            losses["distogram_trajectory"] = distogram_nll.mean()
            total += 10.0 * distogram_nll[-1] + 5.0 * distogram_nll.mean()

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

        # VQ losses
        if c.codebook_size and not c.is_decoder:
            cl = result["codebook_losses"]
            losses.update(cl)
            if not c.state:
                total += c.codebook_loss_scale * (cl["codebook"] + cl["unassigned"])
            total += c.codebook_loss_scale * c.codebook_b * cl["commitment"]

        # additional denoising losses
        if (c.input_diffusion or c.latent_diffusion) and c.latent_loss_scale:
            # sqrt_alpha = jnp.sqrt(1 - data["time"] ** 2 + 1e-6)[:, None]
            # sqrt_snr = sqrt_alpha / (data["time"] + 1e-3)[:, None]
            # noise_gt = (data["latent"] / sqrt_alpha - data["clean_latent"]) * sqrt_snr
            # noise_gt = jax.lax.stop_gradient(noise_gt)
            # predicted_latent = result["predicted_latent"]
            # noise_predicted = (jax.lax.stop_gradient(data["latent"]) / sqrt_alpha - predicted_latent) * sqrt_snr
            # raw_loss = ((noise_gt - noise_predicted) ** 2).mean(axis=-1)
            # unweighted loss
            raw_loss = ((data["clean_latent"] - result["predicted_latent"]) ** 2).mean(axis=-1)
            if c.vp_diffusion:
                weighted_loss = (jnp.where(mask, raw_loss, 0) * base_weight).sum()
            else:
                time = jnp.maximum(data["time"], 1e-2)
                raw_loss = raw_loss * (1 + time ** 2) / jnp.maximum(time ** 2, 1e-6)
                weighted_loss = (jnp.where(mask, raw_loss, 0) * base_weight).sum()
            losses["latent"] = weighted_loss
            total += c.latent_loss_scale * weighted_loss
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

    def __call__(self, local, pos,
                 resi, chain, batch, mask,
                 sup_neighbours):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the decoder block
                features, pos, sup_distogram = block(c)(
                    features, pos,
                    resi, chain, batch, mask,
                    sup_neighbours)
                trajectory_output = pos
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), (trajectory_output, sup_distogram)
            return _inner
        decoder_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(decoder_block)))
        (local, pos), (trajectory, sup_distogram) = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
            sup_distogram = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], sup_distogram))
        return local, pos, trajectory, sup_distogram

class SemiEquivariantDecoderStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "decoder_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos,
                 resi, chain, batch, mask,
                 sup_neighbours):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the decoder block
                features, pos, sup_distogram = block(c)(
                    features, pos,
                    resi, chain, batch, mask,
                    sup_neighbours)
                trajectory_output = pos
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), (trajectory_output, sup_distogram)
            return _inner
        decoder_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(decoder_block)))
        pos = structure_augmentation(pos / c.sigma_data, batch, mask) * c.sigma_data
        (local, pos), (trajectory, sup_distogram) = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
            sup_distogram = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], sup_distogram))
        return local, pos, trajectory, sup_distogram

# nonequivariant decoder
class NonEquivariantDecoderStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "decoder_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos,
                 resi, chain, batch, mask,
                 sup_neighbours):
        c = self.config
        def stack_inner(block):
            def _inner(localpos):
                # run the decoder block
                localpos, sup_distogram = block(c)(
                    localpos,
                    resi, chain, batch, mask,
                    sup_neighbours)
                # return features & positions for the next block
                # and positions to construct a trajectory
                return localpos, (localpos, sup_distogram)
            return _inner
        decoder_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(decoder_block)))
        augment_pos = structure_augmentation(pos / c.sigma_data, batch, mask)
        localpos = Linear(c.local_size, bias=False, initializer="linear")(
            jnp.concatenate((local, augment_pos.reshape(pos.shape[0], -1)), axis=-1)
        )
        localpos, (trajectory, sup_distogram) = stack(localpos)
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
            sup_distogram = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], sup_distogram))
        pos_project = Linear(5 * 3, bias=False)
        trajectory = hk.LayerNorm([-1], True, True)(trajectory)
        trajectory = pos_project(trajectory).reshape(
            trajectory.shape[0], trajectory.shape[1], 5, 3)
        trajectory *= c.sigma_data
        pos = trajectory[-1]
        return local, pos, trajectory, sup_distogram

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
    def inner(pos, dmap, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
        if dmap is not None:
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

def convolution_decoder_pair_features(c, center_features=False):
    def inner(pos, dmap, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        if dmap is not None:
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(pair_mask[..., None],
                        distance_rbf(dmap), 0))
        pos = Vec3Array.from_array(pos)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        local_neighbourhood = jnp.concatenate((
            jnp.repeat(pos.to_array()[:, None], neighbours.shape[1], axis=1),
            pos.to_array()[neighbours]
        ), axis=-2) / c.sigma_data
        if center_features:
            local_neighbourhood -= pos.to_array()[:, None, 1, None]
        local_neighbourhood = local_neighbourhood.reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            local_neighbourhood)
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

def hybrid_decoder_pair_features(c, center_features=False):
    def inner(pos, dmap, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        if dmap is not None:
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(pair_mask[..., None],
                        distance_rbf(dmap), 0))
        pos = Vec3Array.from_array(pos)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        local_neighbourhood = jnp.concatenate((
            jnp.repeat(pos.to_array()[:, None], neighbours.shape[1], axis=1),
            pos.to_array()[neighbours]
        ), axis=-2) / c.sigma_data
        center_neighbourhood = local_neighbourhood -  pos.to_array()[:, None, 1, None]
        center_neighbourhood = (Vec3Array.from_array(center_neighbourhood).normalized()).to_array()
        local_neighbourhood = local_neighbourhood.reshape(*neighbours.shape, -1)
        center_neighbourhood = center_neighbourhood.reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            jnp.concatenate((local_neighbourhood, center_neighbourhood), axis=-1))
        # pseudo rotation features
        # local_vectors = (pos - pos[:, 1, None]).normalized().to_array()
        # pseudo_rot = jnp.einsum("...jc,...nkc->...njk", local_vectors, local_vectors[neighbours])
        # pseudo_rot = pseudo_rot.reshape(*neighbours.shape, -1)
        dirs = (pos[:, None, :, None] - pos[neighbours, None, :]).to_array()
        dirs = dirs.reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(dirs)

        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

def nonequivariant_decoder_pair_features(c):
    def inner(local, dmap, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        if dmap is not None:
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(pair_mask[..., None],
                        distance_rbf(dmap), 0))
        pair += Linear(c.pair_size, bias=False)(local)[:, None]
        pair += Linear(c.pair_size, bias=False)(local)[neighbours]
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1],
                   activation=jax.nn.gelu,
                   final_init="linear")(pair)
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

class VQ(hk.Module):
    def __init__(self, codebook_size=4096,
                 affine=None,
                 mapped_axes=None,
                 name: str | None = "vq"):
        super().__init__(name)
        self.codebook_size = codebook_size
        self.affine = affine
        self.mapped_axes = mapped_axes if mapped_axes else []

    def __call__(self, features, mask):
        def replace_gradient(x, y):
            return jax.lax.stop_gradient(x) + y - jax.lax.stop_gradient(y)
        codebook = 10.0 * hk.get_parameter("codebook",
                                    (self.codebook_size, features.shape[-1]),
                                    init=hk.initializers.RandomUniform(-0.1, 0.1))
        if self.affine:
            codebook_mean = hk.get_parameter("codebook_mean",
                                             (1, features.shape[-1]),
                                             init=hk.initializers.Constant(0.0))
            codebook_scale = hk.get_parameter("codebook_scale",
                                              (1, features.shape[-1]),
                                              init=hk.initializers.Constant(0.0))
            codebook_scale = jnp.exp(codebook_scale)
            codebook = codebook_mean + codebook_scale * codebook
        distance = ((features[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1)
        distance = jnp.where(mask[:, None], distance, jnp.inf)
        assign_fwd = jnp.argmin(distance, axis=1)
        assign_rev = jnp.argmin(distance, axis=0)
        features_fwd = codebook[assign_fwd]
        features_rev = features[assign_rev]
        out_features = replace_gradient(features_fwd, features)
        codebook_loss = ((features_fwd - jax.lax.stop_gradient(features)) ** 2).mean(axis=-1)
        codebook_loss = jnp.where(mask, codebook_loss, 0).sum() / jnp.maximum(mask.sum(), 1)
        commitment_loss = ((jax.lax.stop_gradient(features_fwd) - features) ** 2).mean(axis=-1)
        commitment_loss = jnp.where(mask, commitment_loss, 0).sum() / jnp.maximum(mask.sum(), 1)
        assignment_count = jnp.zeros((self.codebook_size,),
                                     dtype=jnp.float32).at[assign_fwd].add(mask)
        # if we are mapping over one or more axes
        # sum assignment_count over all of them
        local_assignment_count = assignment_count
        if (not hk.running_init()):
            for axis in self.mapped_axes:
                assignment_count = jax.lax.psum(assignment_count, axis)
        assignment_mask = local_assignment_count < 1
        unassigned_loss = assignment_mask * ((codebook - jax.lax.stop_gradient(features_rev)) ** 2).mean(axis=-1)
        unassigned_loss = unassigned_loss.sum() / jnp.maximum(assignment_mask.sum(), 1)
        losses = dict(
            codebook=codebook_loss,
            commitment=commitment_loss,
            unassigned=unassigned_loss,
            unassigned_percent=(assignment_count > 0).mean()
        )
        return out_features, assign_fwd, losses

class FSQ(hk.Module):
    def __init__(self, name: str | None = "fsq"):
        super().__init__(name)

    def __call__(self, features):
        downcast = Linear(7, bias=False)(features)
        half_l = (3 * (1 - 1e-3)) / 2
        offset = 0.5
        shift = jnp.tan(offset / half_l)
        downcast = jnp.tanh(downcast + shift) * half_l - offset
        rounded = jnp.round(jax.lax.stop_gradient(downcast))
        rounded = jax.lax.stop_gradient(rounded) + (downcast - jax.lax.stop_gradient(downcast))
        upcast = Linear(features.shape[-1], bias=False)(rounded / 2)
        return upcast, rounded

class VQState(hk.Module):
    def __init__(self, codebook_size=4096, gamma=0.99,
                 mapped_axes=None,
                 name: str | None = "vq"):
        super().__init__(name)
        self.codebook_size = codebook_size
        self.gamma = gamma
        self.mapped_axes = mapped_axes if mapped_axes else []

    def __call__(self, features, mask):
        def replace_gradient(x, y):
            return jax.lax.stop_gradient(x) + y - jax.lax.stop_gradient(y)
        codebook = hk.get_state("codebook",
                                (self.codebook_size, features.shape[-1]),
                                init=hk.initializers.RandomNormal())
        prev_codebook = codebook
        count = hk.get_state("count",
                             (self.codebook_size,),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(0.0))
        avg = hk.get_state("avg",
                             (self.codebook_size,),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(0.0))
        distance = ((features[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1)
        distance = jnp.where(mask[:, None], distance, jnp.inf)
        assign_fwd = jnp.argmin(distance, axis=1)
        assign_rev = jnp.argmin(distance, axis=0)
        features_fwd = codebook[assign_fwd]
        features_rev = features[assign_rev]
        out_features = replace_gradient(features_fwd, features)
        total = jnp.maximum(mask.sum(), 1)
        codebook_loss = ((features_fwd - jax.lax.stop_gradient(features)) ** 2).mean(axis=-1)
        codebook_loss = jnp.where(mask, codebook_loss, 0).sum() / total
        commitment_loss = ((jax.lax.stop_gradient(features_fwd) - features) ** 2).mean(axis=-1)
        commitment_loss = jnp.where(mask, commitment_loss, 0).sum() / total
        assignment_count = jnp.zeros((self.codebook_size,),
                                     dtype=jnp.float32).at[assign_fwd].add(mask)
        # if we are mapping over one or more axes
        # sum assignment_count over all of them
        if not hk.running_init():
            for axis in self.mapped_axes:
                assignment_count = jax.lax.psum(assignment_count, axis)
        assignment_mask = assignment_count < 1
        count = (1 - self.gamma) * assignment_count + self.gamma * count
        avg = (1 - self.gamma) * assignment_count / total  + self.gamma * avg
        alpha = jnp.exp(-avg * codebook.shape[0] * 10 / (1 - self.gamma) - 1e-3)
        assigned_update = self.gamma * codebook + (1 - self.gamma) * jnp.zeros_like(codebook).at[assign_fwd].add(features)
        assigned_update /= jnp.maximum(count[:, None], 1)
        unassigned_update = (1 - alpha)[:, None] * codebook + alpha[:, None] * features_rev
        codebook = jnp.where(assignment_mask[:, None], assigned_update, unassigned_update)
        # update codebook with average update over replicas
        codebook_delta = prev_codebook - codebook
        if not hk.running_init():
            for axis in self.mapped_axes:
                codebook_delta = jax.lax.pmean(codebook_delta, axis)
        codebook = prev_codebook + codebook_delta
        losses = dict(
            commitment=commitment_loss,
            unassigned_percent=(assignment_count > 0).mean()
        )
        state_update = dict(count=count, avg=avg, codebook=codebook)
        # hk.set_state("count", count)
        # hk.set_state("avg", avg)
        # hk.set_state("codebook", codebook)
        return out_features, assign_fwd, losses, state_update

class QuickDistogram(hk.Module):
    def __init__(self, config, name: Optional[str] = "quick_distogram"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, resi, chain, batch, neighbours):
        dcode_left = Linear(32, bias=False)(features)
        dcode_right = Linear(32, bias=False)(features)
        dcode = dcode_left[:, None] + dcode_right[neighbours]
        pair_resi = sequence_relative_position(32, one_hot=True)(
            resi, chain, batch, neighbours)
        dcode += Linear(32, bias=False)(pair_resi)
        dcode = hk.LayerNorm([-1], True, True)(dcode)
        distogram_logits = MLP(64, 16, depth=2, activation=jax.nn.gelu, final_init="zeros")(dcode)
        distogram_logits = jax.nn.log_softmax(distogram_logits, axis=-1)
        start = 0.0
        stop = 22.0
        step = (stop - start) / 16
        bin_centers = jnp.arange(16) * step + step / 2
        dmap = (jax.nn.softmax(distogram_logits, axis=-1) * bin_centers).sum(axis=-1)
        return distogram_logits, dmap

class InnerDistogram(hk.Module):
    def __init__(self, config, name: Optional[str] = "inner_distogram"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, resi, chain, batch, neighbours):
        dcode = Linear(16 * 8, bias=False)(features).reshape(features.shape[0], 8, 16)
        dgate = Linear(16 * 8, bias=False)(features).reshape(features.shape[0], 8, 16)
        dgate = jax.nn.gelu(dgate)
        weight = hk.get_parameter("inner_weight", (8, 8, 16), init=hk.initializers.Constant(0.0))
        logits = jnp.einsum("iax,jbx,abx->ijx", dcode, dgate, weight)
        logits = (logits + jnp.swapaxes(logits, 0, 1)) / 2
        logits = jax.nn.log_softmax(logits, axis=-1)
        start = 0.0
        stop = 22.0
        step = (stop - start) / 16
        bin_centers = jnp.arange(16) * step + step / 2
        dmap = (jax.nn.softmax(logits, axis=-1) * bin_centers).sum(axis=-1)
        return logits, dmap

class NonEquivariantDecoderBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features,
                 resi, chain, batch, mask,
                 sup_neighbours=None,
                 pos_gt=None):
        c = self.config
        distogram_block = {
            "mlp": QuickDistogram,
            "inner": InnerDistogram,
            "none": None
        }[c.distogram_block]
        dist = distogram_block(c)
        distogram_logits, dmap = dist(features, resi, chain, batch, None)
        index = axis_index(sup_neighbours, axis=0)
        distogram_logits = distogram_logits[index[:, None], sup_neighbours]
        dmap_neighbours = extract_dmap_neighbours(
            count=32)(
                jax.lax.stop_gradient(dmap),
                resi, chain, batch, mask)

        pair, pair_mask = nonequivariant_decoder_pair_features(c)(
            hk.LayerNorm([-1], True, True)(features),
            dmap, dmap_neighbours,
            resi, chain, batch, mask)
        features += SparseAttention(size=c.key_size,
                                    heads=c.heads)(
            hk.LayerNorm([-1], True, True)(features),
            pair, dmap_neighbours, pair_mask)
        # global update of local features
        features += NonEquivariantDecoderUpdate(c)(
            hk.LayerNorm([-1], True, True)(features),
            chain, batch, mask)
        return features, distogram_logits

class SemiEquivariantDecoderBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos,
                 resi, chain, batch, mask,
                 sup_neighbours=None,
                 pos_gt=None):
        c = self.config
        distogram_block = {
            "mlp": QuickDistogram,
            "inner": InnerDistogram,
            "none": None
        }[c.distogram_block]
        dist = distogram_block(c)
        distogram_logits, dmap = dist(features, resi, chain, batch, None)
        index = axis_index(sup_neighbours, axis=0)
        distogram_logits = distogram_logits[index[:, None], sup_neighbours]
        dmap_neighbours = extract_dmap_neighbours(
            count=32)(
                jax.lax.stop_gradient(dmap),
                resi, chain, batch, mask)

        current_neighbours = extract_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=32)(
                Vec3Array.from_array(pos),
                resi, chain, batch, mask)

        pair, pair_mask = hybrid_decoder_pair_features(c, center_features=False)(
            pos, dmap, current_neighbours,
            resi, chain, batch, mask)
        features += SparseSemiEquivariantPointAttention(c.key_size, c.heads)(
            hk.LayerNorm([-1], True, True)(features),
            pair, pos / c.sigma_data,
            current_neighbours, pair_mask)
        if distogram_block is not None:
            # FIXME: was decoder pair features
            pair, pair_mask = hybrid_decoder_pair_features(c)(
                pos, dmap, dmap_neighbours,
                resi, chain, batch, mask)
            features += SparseSemiEquivariantPointAttention(c.key_size, c.heads)(
                hk.LayerNorm([-1], True, True)(features),
                pair, pos / c.sigma_data,
                dmap_neighbours, pair_mask)
        # global update of local features
        features += NonEquivariantDecoderUpdate(c)(
            hk.LayerNorm([-1], True, True)(features),
            chain, batch, mask)
        local_norm = hk.LayerNorm([-1], True, True)(features)
        # update positions
        split_scale_out = c.sigma_data
        pos = semiequivariant_update_positions(pos, local_norm,
                                               scale=split_scale_out,
                                               symm=c.symm)
        return features, pos.astype(local_norm.dtype), distogram_logits

class DecoderBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos,
                 resi, chain, batch, mask,
                 sup_neighbours=None,
                 pos_gt=None):
        c = self.config
        distogram_block = {
            "mlp": QuickDistogram,
            "inner": InnerDistogram,
            "none": None
        }[c.distogram_block]
        if distogram_block is not None:
            dist = distogram_block(c)
            if c.teacher_forcing_style:
                distogram_logits, _ = dist(features, resi, chain, batch, sup_neighbours)
                cb = Vec3Array.from_array(pos_gt[:, -1])
                dmap = (cb[:, None] - cb[None, :]).norm()
            else:
                distogram_logits, dmap = dist(features, resi, chain, batch, None)
                index = axis_index(sup_neighbours, axis=0)
                distogram_logits = distogram_logits[index[:, None], sup_neighbours]
            dmap_neighbours = extract_dmap_neighbours(
                count=32)(
                    jax.lax.stop_gradient(dmap),
                    resi, chain, batch, mask)
        else:
            # placeholder
            distogram_logits = jnp.zeros((1,), dtype=jnp.float32)
            dmap = None

        current_neighbours = extract_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=32)(
                Vec3Array.from_array(pos),
                resi, chain, batch, mask)

        pair, pair_mask = decoder_pair_features(c)(
            pos, dmap, current_neighbours,
            resi, chain, batch, mask)
        features += SparseStructureAttention(c)(
                hk.LayerNorm([-1], True, True)(features),
                pos / c.sigma_data, pair, pair_mask,
                current_neighbours, resi, chain, batch, mask)
        if distogram_block is not None:
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
        return features, pos.astype(local_norm.dtype), distogram_logits

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

class NonEquivariantDecoderUpdate(hk.Module):
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, chain, batch, mask):
        c = self.config
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

def semiequivariant_update_positions(pos, local_norm, scale=10.0, symm=None):
    pos_update = scale * Linear(
        pos.shape[-2] * 3, initializer=init_zeros(),
        bias=False)(local_norm)
    # potentially symmetrize position update
    pos_update = pos_update.reshape(*pos_update.shape[:-1], -1, 3)
    if symm is not None:
        pos_update = symm(pos_update)

    # project updated pos to global coordinates
    pos += pos_update
    return pos
