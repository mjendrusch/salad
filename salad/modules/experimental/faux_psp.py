import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from salad.aflib.model.geometry import Vec3Array, Rigid3Array
from salad.aflib.model.layer_stack import layer_stack
from salad.aflib.model.all_atom_multimer import get_atom14_mask
from salad.modules.utils.alphafold_loss import violation_loss

from salad.modules.basic import Linear, MLP, init_glorot
from salad.modules.utils.geometry import positions_to_ncacocb, get_spatial_neighbours, distance_one_hot
from salad.modules.geometric import (
    pair_vector_features, distance_features, rotation_features, extract_aa_frames,
    SparseStructureAttention, SparseInvariantPointAttention)
from salad.modules.utils.dssp import assign_dssp
from salad.modules.noise_schedule_benchmark import update_positions

def stogra(c, x):
    if c.gen:
        return x
    return jax.lax.stop_gradient(x)

def drop_token(c, x, p=0.1):
    r"""Drop entire latent tokens (zero everything)."""
    if c.eval:
        return x
    mask = jax.random.bernoulli(hk.next_rng_key(), p, (x.shape[0],)) > 0
    return jnp.where(mask[:, None], 0, x)

def drop_features(c, x, p=0.25):
    r"""Drop individual latent features (zero random)."""
    if c.eval:
        return x
    mask = jax.random.bernoulli(hk.next_rng_key(), p, x.shape) > 0
    return jnp.where(mask, 0, x)

# TODO: Make version with AA prediction instead of embedding
# stopgradient of predicted AA is then fed into the decoder
# -> close to a structure predictor and hopefully less
#    prone to generating shit structures
class StructureEncoding(hk.Module):
    def __init__(self, config, name = "structure_encoding"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        encoder = self.get_encoder()
        decoder = self.get_decoder()
        diagnostics = DiagnosticsHead(c.diagnostics)

        latent = encoder(data)
        # logits / probabilities as structure encoding
        if c.encoder.kind == "AAEncoder":
            mask = data["residue_mask"]
            mask_count = jnp.maximum(mask.sum(), 1)
            gt_aa = jax.nn.one_hot(data["aa_gt"], 20, axis=-1)
            encoder_aa_loss = -jnp.where(
                mask, (latent * gt_aa).sum(axis=-1), 0).sum() / mask_count
            if c.sample:
                sampled_one_hot = jax.nn.one_hot(
                    jax.random.categorical(hk.next_rng_key(), (1 / c.temperature) * latent),
                    20, axis=-1)
                N = jax.random.randint(hk.next_rng_key(), (), 0, 100)
                sampled_n_mask = jnp.arange(100) <= N
                sample = jax.random.categorical(
                    hk.next_rng_key(), (1 / c.temperature) * latent[:, None],
                    shape=list(latent.shape[:-1]) + [100])
                sample = jnp.where(sampled_n_mask[None, :], sample, 20)
                sampled_n_hot = jax.nn.one_hot(sample, 20, axis=-1).sum(axis=-2)
                sampled_n_hot /= jnp.maximum(
                    1, sampled_n_hot.sum(axis=-1, keepdims=True))
                inflate_1 = jax.random.bernoulli(hk.next_rng_key(), p=0.25, shape=())
                latent = jnp.where(inflate_1, sampled_one_hot, sampled_n_hot)
                latent = sampled_one_hot
            else:
                latent = jax.lax.stop_gradient(jnp.exp(latent))
            if c.hard:
                latent = jax.nn.one_hot(jnp.argmax(latent, axis=-1), 20, axis=-1)
        diag_result = diagnostics(jax.lax.stop_gradient(latent))
        true_latent = None
        if not hk.running_init() and c.drop_latent:
            # introducing noise to latents during training encourages
            # the error module to predict errors depending on the noise
            # level. This is undesirable behaviour. The error model should
            # be able to quantify the error / realism of a prediction
            # independent of the noise level.
            # This could be achieved by corrupting protein latents in different ways.
            # In short, error prediction with noisy latents is too simple
            # as a task to result in a reasonable error predictor.
            #
            # alternatives:
            # - noise the input to the encoder instead
            # - noise the pre-tanh latents
            # - replace a random percentage of latents with random values between 0 and 1
            mean = c.noise_mean or 0.0
            std = c.noise_std or 1.0
            noise_level = jnp.exp(mean + std * jax.random.normal(hk.next_rng_key(), ()))
            noise = noise_level * jax.random.normal(hk.next_rng_key(), latent.shape)
            true_latent = latent
            latent += noise
            p_drop = jax.random.uniform(hk.next_rng_key(), (), minval=0.1, maxval=0.5)
            latent = drop_token(c, latent, p=p_drop)
            # latent = drop_features(c, latent, p=0.2)
        if not hk.running_init() and c.mix_latent:
            mix_target = jax.random.uniform(
                hk.next_rng_key(), latent.shape, minval=-1.0, maxval=1.0)
            mix_alpha = jax.random.uniform(hk.next_rng_key(), ())
            p_mix = jax.random.uniform(hk.next_rng_key(), (), minval=0, maxval=1.0)
            mix_mask = jax.random.bernoulli(hk.next_rng_key(), p_mix, (latent.shape[0],))
            mixed = (1 - mix_alpha) * latent + mix_alpha * mix_target
            latent = jnp.where(mix_mask[:, None], mixed, latent)
        diag_total, diag_losses = diagnostics.loss(diag_result, data)
        result = decoder(latent, data)
        result["latent"] = latent
        if true_latent is not None:
            result["latent_gt"] = true_latent
        total, losses = decoder.loss(result, data)
        losses.update(diag_losses)
        if c.encoder.kind == "AAEncoder":
            losses["encoder_aa"] = encoder_aa_loss
            total += encoder_aa_loss
        out = dict(
            results=result,
            losses=losses
        )
        total += diag_total
        return total, out

    def get_encoder(self):
        c = self.config.encoder
        return globals()[c.kind](c)

    def get_decoder(self):
        c = self.config.decoder
        return globals()[c.kind](c)
    
class StructureHal(StructureEncoding):
    def __call__(self, latent):
        c = self.config
        encoder = self.get_encoder()
        decoder = self.get_decoder()
        diagnostics = DiagnosticsHead(c.diagnostics)

        data = self.init_data(latent)
        result = hk.remat(decoder)(latent, data)
        result["latent"] = latent
        gen_data = self.make_gen_data(result)
        # encoded_latent = encoder(gen_data)
        distogram = result["distogram"][..., :20]
        distogram_entropy = -(jax.nn.softmax(distogram, axis=-1) * distogram).sum(axis=-1)
        contact_loss = jnp.sort(distogram_entropy, axis=1)[:, :2].sum(axis=1).mean()
        # dist = (distogram * jnp.linspace(0, 22.0, 64)).sum(axis=1)
        pae = (jnp.exp(result["pae"]) * jnp.linspace(0, 40.0, 64)).sum(axis=1).mean()
        plddt = (jnp.exp(result["plddt"]) * jnp.linspace(0, 100.0, 64)).sum(axis=1).mean()
        dssp = jnp.exp(result["dssp"]) # LEH
        non_loop = (1 - dssp[..., 0]).sum()
        # latent_harmonic = ((encoded_latent - jax.lax.stop_gradient(latent)) ** 2).mean()
        ca = Vec3Array.from_array(result["trajectory"][-1][:, 1])
        dist = (ca[:, None] - ca[None, :]).norm()
        resi = data["residue_index"]
        rdist = abs(resi[:, None] - resi[None, :])
        next_res = rdist == 1
        # next_res_distance = (next_res * (dist - 3.81) ** 2).sum() / next_res.sum() 
        far = rdist > 16
        # clash = (far * jax.nn.relu(6 - dist) ** 2).sum() / far.sum()
        total = pae + contact_loss + (100 - plddt)# + 0.1 * distogram_entropy - 0.1 * non_loop# + 100 * clash + 10 * next_res_distance#distogram_entropy#pae# + (100 - plddt)# + latent_harmonic# + (100 - plddt)
        return total, result

    def init_data(self, latent):
        return dict(
            all_atom_positions=jnp.zeros((latent.shape[0], 14, 3), dtype=jnp.float32),
            all_atom_mask=jnp.zeros((latent.shape[0], 14), dtype=jnp.bool_),
            residue_index=jnp.arange(latent.shape[0], dtype=jnp.int32),
            chain_index=jnp.zeros((latent.shape[0],), dtype=jnp.int32),
            batch_index=jnp.zeros((latent.shape[0],), dtype=jnp.int32),
            aa_gt=jnp.zeros((latent.shape[0],), dtype=jnp.int32),
            residue_mask=jnp.ones((latent.shape[0],), dtype=jnp.int32)
        )

    def make_gen_data(self, result):
        size = result["trajectory"][-1].shape[0]
        return dict(
            all_atom_positions=result["trajectory"][-1],
            all_atom_mask=jnp.zeros((size, 14), dtype=jnp.bool_).at[:4].set(True),
            residue_index=jnp.arange(size, dtype=jnp.int32),
            chain_index=jnp.zeros((size,), dtype=jnp.int32),
            batch_index=jnp.zeros((size,), dtype=jnp.int32),
            aa_gt=jnp.zeros((size,), dtype=jnp.int32),
            residue_mask=jnp.ones((size,), dtype=jnp.int32)
        )


class DiagnosticsHead(hk.Module):
    def __init__(self, config, name = "diagnostics"):
        super().__init__(name)
        self.config = config

    def __call__(self, local):
        c = self.config
        result = dict()
        # result["absolute_order"] = jax.nn.log_softmax(MLP(
        #     local.shape[-1] * 2,
        #     c.crop_size,
        #     activation=jax.nn.gelu,
        #     final_init="zeros")(local))
        result["relative_order"] = jax.nn.log_softmax(
            jnp.einsum("...aik,...bik->...abk",
                       Linear(c.pair_size * 66, bias=True)(local).reshape(-1, c.pair_size, 66),
                       Linear(c.pair_size * 66, bias=True)(local).reshape(-1, c.pair_size, 66)))
        result["contact"] = xent_log(
            jnp.einsum("...ai,...bi->...ab",
                       Linear(c.pair_size, bias=True)(local),
                       Linear(c.pair_size, bias=True)(local)))
        result["distogram"] = jax.nn.log_softmax(
            jnp.einsum("...aik,...bik->...abk",
                       Linear(c.pair_size * 64, bias=True)(local).reshape(-1, c.pair_size, 64),
                       Linear(c.pair_size * 64, bias=True)(local).reshape(-1, c.pair_size, 64)))
        return result

    def loss(self, result, data):
        c = self.config
        losses = dict()
        total = 0.0
        mask = data["residue_mask"]
        cb = positions_to_ncacocb(data["all_atom_positions"][:, :14])[:, 4]
        pair_mask = mask[:, None] * mask[None, :]
        pair_count = jnp.maximum(pair_mask.sum(), 1)
        gt_dist = jnp.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
        gt_contacts = gt_dist < 8
        gt_contacts = jax.nn.one_hot(gt_contacts.astype(jnp.int32), 2)
        if c["contact"]:
            contact_nll = -jnp.where(
                pair_mask, (result["contact"] * gt_contacts).sum(axis=-1), 0).sum() / pair_count
            losses["contact_diagnostic"] = contact_nll
            total += c["contact"] * contact_nll
        if c["distogram"]:
            gt_distogram = distance_one_hot(gt_dist)
            distogram_nll = -jnp.where(
                pair_mask, (result["distogram"] * gt_distogram).sum(axis=-1), 0).sum() / pair_count
            losses["distogram_diagnostic"] = distogram_nll
            total += c["distogram"] * distogram_nll
        if c["relative_order"]:
            resi = data["residue_index"]
            chain = data["chain_index"]
            same_chain = chain[:, None] == chain[None, :]
            gt_order = jnp.clip(resi[:, None] - resi[None, :], -32, 32) + 32
            gt_order = jnp.where(same_chain, gt_order, 65)
            gt_order = jax.nn.one_hot(gt_order, 66, axis=-1)
            order_nll = -jnp.where(
                pair_mask, (result["relative_order"] * gt_order).sum(axis=-1), 0).sum() / pair_count
            losses["relative_order_diagnostic"] = order_nll
            total += c["relative_order"] * order_nll
        return total, losses

class UnorderedEncoder(hk.Module):
    def __init__(self, config, name = "unordered_encoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        encoder_stack = EncoderStack(c, UnorderedEncoderBlock)
        local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data)
        local = encoder_stack(local, pos, neighbours, resi, chain, batch, mask)
        latent = jax.lax.tanh(Linear(c.latent_size, bias=False)(local))
        return latent

    def prepare_features(self, data):
        c = self.config
        pos = data["all_atom_positions"][:, :14]
        pos_mask = data["all_atom_mask"][:, :14]
        mask = pos_mask[:, 1]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]

        pos = positions_to_ncacocb(pos)
        if not c.eval:
            pos += c.noise_level * jax.random.normal(hk.next_rng_key(), pos.shape)
        ca = pos[:, 1]
        neighbours = get_spatial_neighbours(c.num_neighbours)(
            Vec3Array.from_array(ca), batch, mask)
        pair, pair_mask = unordered_pair_features(size=c.pair_size)(pos, neighbours, batch, mask)
        pair_weight = jax.nn.gelu(Linear(pair.shape[-1], bias=False, initializer="relu")(pair))
        local = jnp.where(pair_mask[..., None], pair * pair_weight, 0).sum(axis=1)
        local = hk.LayerNorm([-1], True, True)(local)
        return local, pos, neighbours, resi, chain, batch, mask

class AAEncoder(UnorderedEncoder):
    def __call__(self, data):
        c = self.config
        encoder_stack = EncoderStack(c, UnorderedEncoderBlock)
        local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data)
        local = encoder_stack(local, pos, neighbours, resi, chain, batch, mask)
        latent = jax.nn.log_softmax(Linear(20, bias=False)(local))
        return latent

class UnorderedEncoderBlock(hk.Module):
    def __init__(self, config, name = "unordered_encoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, neighbours, resi, chain, batch, mask):
        c = self.config
        pair, pair_mask = unordered_pair_features(size=c.pair_size)(
            pos, neighbours, batch, mask)
        local += SparseStructureAttention(c)(
            local, pos, pair, pair_mask,
            neighbours, resi, chain, batch, mask)
        local += Update(c, name="update")(local)
        return local

class EncoderStack(hk.Module):
    def __init__(self, config, block, name = "encoder_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, neighbours, resi, chain, batch, mask):
        c = self.config
        def iter(local):
            return hk.remat(self.block(c))(
                local, pos, neighbours, resi, chain, batch, mask)
        local = layer_stack(c.depth)(iter)(local)
        if c.encoding_noise:
            local = hk.LayerNorm([-1], False, False)(local)
            if not c.eval:
                noise_level = jnp.exp(0.5 + 1.2 * jax.random.normal(hk.next_rng_key(), ()))
                noise = noise_level * jax.random.normal(hk.next_rng_key(), local.shape)
                local = local + noise
            local = hk.LayerNorm([-1], True, True)(local)
        else:
            local = hk.LayerNorm([-1], True, True)(local)
        return local

def unordered_pair_features(size):
    def _inner(pos, neighbours, batch, mask):
        if not isinstance(pos, Vec3Array):
            pos = Vec3Array.from_array(pos)
        pair_mask = (neighbours != -1) * mask[:, None] * mask[neighbours]
        pair = Linear(size, bias=False)(distance_features(pos, neighbours))
        pair += Linear(size, bias=False)(rotation_features(extract_aa_frames(pos)[0], neighbours))
        pair += Linear(size, bias=False)(pair_vector_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(size * 2, size, activation=jax.nn.gelu)(pair)
        return pair, pair_mask
    return _inner

def error_pair_features(size):
    def _inner(pos, mask):
        if not isinstance(pos, Vec3Array):
            pos = Vec3Array.from_array(pos)
        pair = Linear(size, bias=False)(distance_features(pos, None))
        pair += Linear(size, bias=False)(rotation_features(extract_aa_frames(pos)[0], None))
        pair += Linear(size, bias=False)(pair_vector_features(pos, None))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(size * 2, size, activation=jax.nn.gelu)(pair)
        return pair
    return _inner

def bond_angle(x, y, z):
    x = Vec3Array.from_array(x)
    y = Vec3Array.from_array(y)
    z = Vec3Array.from_array(z)
    
    vx = (x - y).normalized()
    vz = (z - y).normalized()
    dot = vx.dot(vz)
    cross = vx.cross(vz).norm()
    result = jnp.arctan2(cross, dot)

    return jax.lax.stop_gradient(result)

def dihedral_angle(*args):
    assert len(args) == 4
    a, b, c, d = [Vec3Array.from_array(c) for c in args]
    u1 = (b - a).normalized()
    u2 = (c - b).normalized()
    u3 = (d - c).normalized()
    result = jnp.arctan2(
        u1.dot(u2.cross(u3)),
        (u1.cross(u2)).dot(u2.cross(u3))
    )
    return jax.lax.stop_gradient(result)

def bin_angle(angle, bins=16):
    offset = 2 * jnp.pi / bins
    bin_centers = np.arange(bins) * offset - jnp.pi
    return jax.nn.one_hot(
        jnp.argmin(abs(angle[..., None] - bin_centers), axis=-1), bins)

def orientogram(pos):
    n = pos[:, 0]
    ca = pos[:, 1]
    cb = pos[:, 4]
    omega = dihedral_angle(
        ca[:, None], cb[:, None],
        cb[None, :], ca[None, :])
    theta = dihedral_angle(
        n[:, None], ca[:, None],
        cb[:, None], cb[None, :])
    phi = bond_angle(
        ca[:, None], cb[:, None], cb[None, :])
    return bin_angle(jnp.stack((omega, theta, phi), axis=-1), bins=24)

def ramagram(pos):
    n = pos[:, 0]
    ca = pos[:, 1]
    c = pos[:, 2]
    phi = bin_angle(jnp.concatenate((
        jnp.zeros((1,), dtype=jnp.float32),
        dihedral_angle(c[:-1], n[1:], ca[1:], c[1:])), axis=0))
    psi = bin_angle(jnp.concatenate((
        dihedral_angle(n[:-1], ca[:-1], c[:-1], n[1:]),
        jnp.zeros((1,), dtype=jnp.float32)), axis=0))
    result = phi[..., :, None] * psi[..., None, :]
    return result

class Decoder(hk.Module):
    def __init__(self, config, name = "decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, latent, data):
        c = self.config

        if c.dssp_short:
            result["dssp_short"] = MLP(c.local_size * 2, 3,
                                       activation=jax.nn.gelu, final_init="zeros")(latent)
        if c.aa_short:
            result["aa_short"] = MLP(c.local_size * 2, 20,
                                       activation=jax.nn.gelu, final_init="zeros")(latent)

        stack = DecoderStack(c)
        def apply_stack(latent, prev=None):
            local, pair, mask = self.prepare_features(latent, data, prev=prev)
            local, pair = stack(local, pair, mask)

            if c.structure_module:
                structure_module = StructureModule(c)
                pos = structure_module.init_pos(local)
                local, pos, trajectory = structure_module(local, pair, pos, mask)
                return local, pair, pos, trajectory
            return local, pair

        def recycling_iter(i, prev):
            result = apply_stack(latent, prev=prev)
            if c.structure_module:
                local, pair, pos, trajectory = result
            else:
                pos = None
                trajectory = None
                local, pair = result
            prev = dict(local=local, pos=pos)
            return prev

        prev = self.init_prev(data)
        if c.recycle and not hk.running_init():
            if c.eval:
                if c.num_recycle is not None:
                    count = c.num_recycle
                else:
                    count = 4
            else:
                count = jax.random.randint(hk.next_rng_key(), (), 0, 4)
            prev = stogra(c,
                hk.fori_loop(0, count, hk.remat(recycling_iter), prev))
        local, pair, mask = self.prepare_features(latent, data, prev=prev)
        local, pair = stack(local, pair, mask)

        result = dict(local=local, pair=pair)
        if c.structure_module:
            structure_module = StructureModule(c)
            pos = structure_module.init_pos(local)
            if c.structure_independent:
                local, result["pos"], result["trajectory"] = structure_module(
                    stogra(c, local),
                    stogra(c, pair),
                    pos, mask)
            else:
                local, result["pos"], result["trajectory"] = structure_module(local, pair, pos, mask)
            result["post_local"] = local

        # local predictions
        result["aa"] = jax.nn.log_softmax(
            MLP(local.shape[-1] * 2, 20,
                activation=jax.nn.gelu, final_init="zeros")(local))
        result["dssp"] = jax.nn.log_softmax(
            MLP(local.shape[-1] * 2, 3,
                activation=jax.nn.gelu, final_init="zeros")(local))
        if c.denoise:
            result["predicted_latent"] = MLP(
                local.shape[-1] * 2, c.latent_size,
                activation=jax.nn.gelu, final_init="zeros")(local)
        if c.ramagram:
            result["ramagram"] = jax.nn.log_softmax(
                MLP(local.shape[-1] * 2, 16 * 16,
                    activation=jax.nn.gelu, final_init="zeros")(local)).reshape(
                        local.shape[0], 16, 16)
        # pair predictions
        result["distogram"] = jax.nn.log_softmax(
            MLP(pair.shape[-1] * 2, 64,
                activation=jax.nn.gelu, final_init="zeros")(pair))
        result["contact"] = xent_log(
            MLP(pair.shape[-1] * 2, 1,
                activation=jax.nn.gelu, final_init="zeros")(pair)[..., 0])
        result["aa_pair"] = jax.nn.log_softmax(
            MLP(pair.shape[-1] * 2, 20 * 20,
                activation=jax.nn.gelu, final_init="zeros")(pair).reshape(*pair.shape[:2], 20, 20))
        if c.orientogram:
            result["orientogram"] = jax.nn.log_softmax(
                MLP(pair.shape[-1] * 2, 24 * 3,
                    activation=jax.nn.gelu, final_init="zeros")(pair).reshape(*pair.shape[:2], 3, 24))
        if c.error:
            error = ErrorModule(c)
            if c.error_independent:
                errors = error(
                    stogra(c, local),
                    stogra(c, pair),
                    stogra(c, pos),
                    mask)
            else:
                errors = error(local, pair, pos, mask)
            result.update(errors)
        if c.symmetrize:
            for name in ("distogram", "contact"):
                result[name] = (result[name] + jnp.swapaxes(result[name], 0, 1)) / 2
        return result

    def init_prev(self, data):
        c = self.config
        if not c.recycle:
            return None
        size = data["residue_index"].shape[0]
        prev = dict(
            pos=jnp.zeros((size, 14, 3), dtype=jnp.float32),
            local=jnp.zeros((size, c.local_size), dtype=jnp.float32)
        )
        return prev

    def prepare_features(self, latent, data, prev=None):
        c = self.config
        resi = data["residue_index"]
        chain = data["chain_index"]
        mask = data["residue_mask"]
        same_chain = chain[:, None] == chain[None, :]
        local = Linear(c.local_size, bias=False)(latent)
        ppair = 0.0
        if prev is not None:
            ppos = prev["pos"]
            pcb = ppos[:, 4]
            pdist = jnp.linalg.norm(pcb[:, None] - pcb[None, :], axis=-1)
            pdist = distance_one_hot(pdist)
            plocal = prev["local"]
            local += MLP(local.shape[-1] * 2, local.shape[-1],
                         activation=jax.nn.gelu, bias=False, final_init="zeros")(plocal)
            ppair = Linear(c.pair_size, bias=False, initializer="zeros")(pdist)
        resi_distance = jnp.clip(resi[:, None] - resi[None, :], -32, 32) + 32
        resi_distance = jnp.where(same_chain, resi_distance, 65)
        resi_distance = jax.nn.one_hot(resi_distance, 66, axis=-1)
        local_pair = jnp.einsum("iac,jac->ija",
                                Linear(16 * 32, bias=True)(local).reshape(-1, 16, 32),
                                Linear(16 * 32, bias=True)(local).reshape(-1, 16, 32))
        pair = jnp.concatenate((resi_distance, local_pair), axis=-1)
        pair = MLP(c.pair_size * 2, c.pair_size, activation=jax.nn.gelu)(pair)
        pair = hk.LayerNorm([-1], True, True)(pair + ppair)
        return local, pair, mask
    
    def loss(self, result, data):
        c = self.config
        losses = dict()
        total = 0.0
        mask = data["residue_mask"]
        batch = data["chain_index"]
        mask_count = jnp.maximum(mask.sum(), 1)
        pos_gt = positions_to_ncacocb(data["all_atom_positions"][:, :14])
        cb = pos_gt[:, 4]
        pair_mask = mask[:, None] * mask[None, :]
        pair_count = jnp.maximum(pair_mask.sum(), 1)
        gt_dist = jnp.linalg.norm(cb[:, None] - cb[None, :], axis=-1)
        gt_dssp, _, _ = assign_dssp(pos_gt[:, :4], batch, mask)
        gt_dssp = jax.nn.one_hot(gt_dssp, 3)
        gt_contacts = gt_dist < 8
        gt_contacts = jax.nn.one_hot(gt_contacts.astype(jnp.int32), 2)
        gt_distogram = distance_one_hot(gt_dist)
        gt_aa = jax.nn.one_hot(data["aa_gt"], 20, axis=-1)
        gt_aa_pair = gt_aa[:, None, :, None] * gt_aa[None, :, None, :]
        losses["mask_count"] = mask_count
        losses["distogram"] = -jnp.where(pair_mask, (result["distogram"] * gt_distogram).sum(axis=-1), 0).sum() / pair_count
        losses["aa_pair"] = -jnp.where(pair_mask, (result["aa_pair"] * gt_aa_pair).sum(axis=(-1, -2)), 0).sum() / pair_count
        losses["contact"] = -jnp.where(pair_mask, (result["contact"] * gt_contacts).sum(axis=-1), 0).sum() / pair_count
        losses["aa"] = -jnp.where(mask, (result["aa"] * gt_aa).sum(axis=-1), 0).sum() / mask_count
        losses["dssp"] = -jnp.where(mask, (result["dssp"] * gt_dssp).sum(axis=-1), 0).sum() / mask_count
        if c.ramagram:
            gt_rama = ramagram(pos_gt)
            losses["ramagram"] = -jnp.where(
                mask, (result["ramagram"] * gt_rama).sum(axis=(-1, -2)), 0).sum() / mask_count
        if c.orientogram:
            gt_orientogram = orientogram(pos_gt)
            losses["orientogram"] = -jnp.where(
                pair_mask, (result["orientogram"] * gt_orientogram).sum(axis=(-1, -2)), 0).sum() / pair_count
        if c.structure_module:
            trajectory = result["trajectory"]
            pos_gt = data["all_atom_positions"][:, :14]
            pos_mask_gt = data["all_atom_mask"][:, :14]
            frames_gt, _ = extract_aa_frames(Vec3Array.from_array(pos_gt))
            frames, _ = extract_aa_frames(Vec3Array.from_array(trajectory.reshape(-1, 14, 3)))
            frames = frames.to_array()
            frames = frames.reshape(*trajectory.shape[:2], *frames.shape[1:])
            frames = Rigid3Array.from_array(frames)
            fapos_gt = frames_gt[:, None, None].apply_inverse_to_point(Vec3Array.from_array(pos_gt)[None, :, :])
            fapos = frames[:, :, None, None].apply_inverse_to_point(Vec3Array.from_array(trajectory)[:, None, :, :])
            raw_ae = (fapos_gt - fapos[-1]).norm()
            ae = jnp.clip((fapos_gt[None] - fapos).norm(), 0.0, 10.0) / 10.0
            fape_mask = mask[:, None, None] * pos_mask_gt[None, :, :]
            raw_error = (raw_ae * fape_mask).sum(axis=-1) / jnp.maximum(fape_mask.sum(axis=-1), 1)
            fape = jnp.where(fape_mask[None], ae, 0).sum(axis=(1, 2, 3))
            fape /= jnp.maximum(fape_mask.sum(), 1)
            fape_trajectory = fape[:-1].mean()
            losses["fape"] = (fape[-1] + 0.5 * fape_trajectory) / 1.5
            losses["fape_final"] = fape[-1]
            losses["fape_trajectory"] = fape_trajectory
        if c.structure_module and c.sidechain_loss:
            # TODO
            pass
        if c.denoise:
            latent_error = ((
                jax.lax.stop_gradient(result["latent_gt"])
                 - result["predicted_latent"]) ** 2).sum(axis=-1)
            losses["denoising"] = jnp.where(mask[:, None], latent_error, 0).sum() / mask_count
        if c.error:
            # pae
            ae_gt = jax.lax.stop_gradient(distance_one_hot(jax.lax.stop_gradient(raw_error), max_distance=40))
            losses["pae"] = -jnp.where(pair_mask, (result["pae"] * ae_gt).sum(axis=-1), 0).sum() / pair_count
            # plddt
            ca_gt = pos_gt[:, 1]
            dist_gt = jnp.linalg.norm(ca_gt[:, None] - ca_gt[None, :], axis=-1)
            ca = Vec3Array.from_array(trajectory[-1, :, 1])
            dist = (ca[:, None] - ca[None, :]).norm()
            dd = abs(dist - dist_gt)
            LOCAL_CUTOFF = 15.0 # angstroms
            local_mask = pair_mask * (dist_gt < LOCAL_CUTOFF)
            ddt = (local_mask * ((dd < 0.5).astype(jnp.float32) +
                                 (dd < 1.0).astype(jnp.float32) +
                                 (dd < 2.0).astype(jnp.float32) +
                                 (dd < 4.0).astype(jnp.float32))).sum(axis=1) / 4
            ddt /= jnp.maximum(local_mask.sum(axis=1), 1)
            lddt_gt = jax.lax.stop_gradient(distance_one_hot(ddt, 0.0, 1.0))
            losses["plddt"] = -jnp.where(mask, (result["plddt"] * lddt_gt).sum(axis=-1), 0).sum() / mask_count
        if "violation" in c.losses:
            res_mask = data["residue_mask"]
            pred_mask = get_atom14_mask(data["aa_gt"]) * res_mask[:, None]
            violation, _ = violation_loss(data["aa_gt"],
                                        data["residue_index"],
                                        trajectory[-1],
                                        pred_mask,
                                        res_mask,
                                        clash_overlap_tolerance=1.5,
                                        violation_tolerance_factor=2.0,
                                        chain_index=data["chain_index"],
                                        batch_index=jnp.zeros_like(data["chain_index"]),
                                        per_residue=False)
            losses["violation"] = violation.mean()
        total = 0.0
        for name, value in losses.items():
            if name in c.losses:
                total += value * c.losses[name]
        return total, losses

def xent_log(x):
    return jnp.stack((
        jax.nn.log_sigmoid(x),
        jax.nn.log_sigmoid(-x)
    ), axis=-1)

class DecoderBlock(hk.Module):
    def __init__(self, config, name = "decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, mask):
        c = self.config
        local += DecoderAttention(c)(
            local, pair, mask)
        local += Update(c)(local)
        if c.pair_update:
            pair += LocalToPair(c)(local, pair, mask)
            pair += Update(c)(pair)
        return local, pair

class ErrorModule(hk.Module):
    def __init__(self, config, name = "module"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, pos, mask):
        c = self.config
        pair += error_pair_features(pair.shape[-1])(pos, mask)
        # error module stack
        def iter(carry):
            local, pair = carry
            return hk.remat(DecoderBlock(c))(
                local, pair, mask)
        local, pair = layer_stack(c.error_depth)(iter)((local, pair))
        local = hk.LayerNorm([-1], True, True)(local)
        pair = hk.LayerNorm([-1], True, True)(pair)
        # predict errors
        result = dict()
        result["plddt"] = jax.nn.log_softmax(Linear(64, initializer="zeros")(local))
        result["pae"] = jax.nn.log_softmax(Linear(64, initializer="zeros")(pair))
        return result

class Update(hk.Module):
    def __init__(self, config, name = "update"):
        super().__init__(name)
        self.config = config

    def __call__(self, x):
        c = self.config
        if c.normalize_input:
            x = hk.LayerNorm([-1], True, True)(x)
        gate = jax.nn.gelu(Linear(
            c.factor * x.shape[-1], bias=False, initializer="relu")(x))
        data = Linear(
            c.factor * x.shape[-1], bias=False, initializer="linear")(x)
        hidden = data * gate
        return Linear(x.shape[-1], bias=False, initializer="zeros")(hidden)

class LocalToPair(hk.Module):
    def __init__(self, config, name = "l2p"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, mask):
        c = self.config
        pair_mask = mask[:, None] * mask[None, :]
        local = hk.LayerNorm([-1], False, False)(local)
        pair = hk.LayerNorm([-1], False, False)(pair)
        # make pseudo-local features
        pair_gate = Linear(2 * c.pair_size, bias=False, initializer="relu")(pair)
        pair_value = Linear(2 * c.pair_size, bias=False)(pair)
        left_gate = jax.nn.gelu(Linear(2 * c.pair_size, bias=False, initializer="relu")(local)[:, None] + pair_gate)
        left_value = Linear(2 * c.pair_size, bias=False)(local)[None, :] + pair_value
        left = jnp.where(pair_mask[..., None], left_gate * left_value, 0).sum(axis=1)
        right_gate = jax.nn.gelu(Linear(2 * c.pair_size, bias=False, initializer="relu")(local)[None, :] + pair_gate)
        right_value = Linear(2 * c.pair_size, bias=False)(local)[:, None] + pair_value
        right = jnp.where(pair_mask[..., None], right_gate * right_value, 0).sum(axis=0)
        paired_pseudo_local = hk.LayerNorm([-1], False, False)(left[:, None] + right[None, :])
        pair_hidden = jnp.concatenate((pair, paired_pseudo_local), axis=-1)
        return Linear(c.pair_size, bias=False, initializer="zeros")(pair_hidden)

class PairWeightedAverage(hk.Module):
    def __init__(self, config, name = "pwa"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, mask):
        c = self.config
        pair_mask = mask[:, None] * mask[None, :]
        local = hk.LayerNorm([-1], True, True)(local)
        value = Linear(c.heads * c.key_size, bias=False)(local)
        attn = Linear(c.heads, bias=False)(pair)
        attn = jnp.where(pair_mask[..., None], attn, -1e9)
        attn = jax.nn.softmax(attn, axis=1)
        attn = jnp.where(pair_mask[..., None], attn, 0)
        out = jnp.einsum("ijh,jhc->ihc", attn, value).reshape(local.shape[0], -1)
        return Linear(local.shape[-1], bias=False, initializer="zeros")(out)

# TODO: figure out if this is the right setup
class FastTriangle(hk.Module):
    def __init__(self, config, name = "triangle"):
        super().__init__(name)
        self.config = config

    def __call__(self, pair, mask):
        c = self.config
        pair = hk.LayerNorm([-1], True, True)(pair)
        left = Linear(pair.shape[-1] // 4, bias=False)(pair)
        right = jnp.swapaxes(Linear(pair.shape[-1] // 4, bias=False)(pair), 0, 1)
        pair = jnp.concatenate((left, right), axis=-1)
        attn = jnp.tanh(Linear(c.heads, bias=True)(pair))
        attn = jnp.where(mask[..., None], attn, 0)
        value = Linear(c.pair.shape[-1] // 2, bias=True)(pair).reshape(*pair.shape[:2], c.heads, -1)
        out = jnp.einsum("ijh,jkhc->ikhc", attn, value)
        out = hk.LayerNorm([-1], False, False)(out.reshape(*pair.shape[:2], -1))
        return Linear(pair.shape[-1], bias=False, initializer="zeros")(out)

class DecoderAttention(hk.Module):
    def __init__(self, config, name = "decoder_attention"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, mask):
        c = self.config
        local = hk.LayerNorm([-1], False, False)(local)
        pair = hk.LayerNorm([-1], False, False)(pair)
        def attn_component(x, h, s):
            return Linear(h * s, bias=True, initializer=init_glorot())(x).reshape(-1, h, s)
        q = (1 / jnp.sqrt(c.key_size + 1e-6)) * hk.LayerNorm([-1], False, False)(
            attn_component(local, c.heads, c.key_size))
        k = hk.LayerNorm([-1], False, False)(
            attn_component(local, c.heads, c.key_size))
        v = attn_component(local, c.heads, c.key_size)
        pair_mask = mask[:, None] * mask[None, :]
        attn = jnp.einsum("iha,jha->ijh", q, k)
        attn += Linear(c.heads, bias=False)(pair)
        attn = jnp.where(pair_mask[..., None], attn, -1e9)
        attn = jax.nn.softmax(attn, axis=1)
        attn = jnp.where(pair_mask[..., None], attn, 0)
        out = jnp.einsum("ijh,jha->iha", attn, v).reshape(local.shape[0], -1)
        if c.pair_update:
            out_pair = Linear(c.key_size, bias=False)(pair)
            out_pair = jnp.einsum("ijh,ija->iha", attn, out_pair).reshape(local.shape[0], -1)
            out = jnp.concatenate((out, out_pair), axis=-1)
        out = Linear(local.shape[-1], bias=False, initializer="zeros")(out)
        return out

class DecoderStack(hk.Module):
    def __init__(self, config, name = "decoder_stack"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, mask):
        c = self.config
        def iter(carry):
            local, pair = carry
            return hk.remat(DecoderBlock(c))(
                local, pair, mask)
        local, pair = layer_stack(c.depth)(iter)((local, pair))
        local = hk.LayerNorm([-1], True, True)(local)
        pair = hk.LayerNorm([-1], True, True)(pair)
        return local, pair

class StructureModule(hk.Module):
    def __init__(self, config, name = "structure_module"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, pos, mask):
        c = self.config
        structure_block = StructureBlock(c)
        def iter_body(carry, i):
            local, pos = carry
            local, pos = structure_block(local, pair, pos, mask)
            return (local, pos), pos
        (local, pos), trajectory = hk.scan(iter_body, (local, pos), None, 8)
        return hk.LayerNorm([-1], True, True)(local), pos, trajectory

    def init_pos(self, local):
        base_frame = jnp.array([[-1.0, 0.0, 0.0],
                                [ 0.0, 0.0, 0.0],
                                [ 0.0, 1.0, 0.0]])
        base_unit = jnp.concatenate((base_frame, np.zeros((11, 3), dtype=jnp.float32)), axis=0)
        return jnp.stack(local.shape[0] * [base_unit], axis=0)

class StructureBlock(hk.Module):
    def __init__(self, config, name = "structure_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, pos, mask):
        c = self.config
        # print("DEBUG:", type(pos), pos.shape)
        mask = mask[:, None] * mask[None, :]
        frames, _ = extract_aa_frames(Vec3Array.from_array(pos * 0.1))
        local += SparseInvariantPointAttention(c.key_size, c.heads, normalize=True)(
            local, pair, frames.to_array(), None, mask)
        local += Update(c)(hk.LayerNorm([-1], True, True)(local))
        local_norm = hk.LayerNorm([-1], True, True)(local)
        pos = update_positions(pos, local_norm)
        return local, pos

# class FauxPSP(hk.Module):
#     def __init__(self, config, name = "faux_psp"):
#         super().__init__(name)
#         self.config = config

#     def __call__(self, data):
#         lws2s = LightWeightS2S(self.config.s2s)
#         predictor = StructurePredictor(self.config.psp)
#         aa = lws2s(data)
#         s2s_loss = self.s2s_loss(aa, data)
#         aa = jax.lax.stop_gradient(jax.nn.softmax(aa))
#         data["aa"] = aa
#         prev = predictor.init_prev(data)
#         def iter_body(prev):
#             result = predictor(data, prev)
#             return dict(
#                 local=result["local"],
#                 pos=result["pos"]
#             )
#         if not hk.running_init():
#             count = jax.random.randint(hk.next_rng_key(), (), 0, 4)
#             prev = hk.fori_loop(0, count, iter_body, prev)
#         result = predictor(data, prev)
#         predictor_loss = self.psp_loss(result, data)
#         ... # TODO
