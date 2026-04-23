import jax
import jax.numpy as jnp
import haiku as hk
from salad.modules.basic import Linear, GatedMLP, MLP, init_zeros
from salad.modules.transformer import drop
from salad.modules.utils.geometry import (
    gaussian_rbf, log_centers, linear_centers,
    freq_embedding, spherical_harmonics, get_neighbours, positions_to_ncacocb)
from salad.modules.utils.augmentation import augment_rotation
from salad.aflib.model.layer_stack import layer_stack
from salad.aflib.model.geometry import Vec3Array, Rot3Array, Rigid3Array

class PottsSequenceModel(hk.Module):
    def __init__(self, config, name = "diffusion_sequence_model"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        # prepare data
        data.update(self.prepare_data(data))
        # data.update(self.apply_diffusion(data))
        # run model
        local, pair, pos, extra_pair, extra_pair_mask, extra_pos, neighbours, mask = self.prepare_features(data)
        mpnn_block = None
        if c.block_type == "extended":
            mpnn_block = ExtendedBlock
        else:
            mpnn_block = MPNNBlock
        def body(carry):
            local, pair, extra_pair = carry
            local, pair, extra_pair = mpnn_block(c)(
                local, pair, pos, extra_pos, extra_pair,
                extra_pair_mask, neighbours, mask)
            return local, pair, extra_pair
        local, pair, extra_pair = layer_stack(c.depth, with_state=False)(hk.remat(body))((local, pair, extra_pair))
        if c.no_renorm:
            local_norm = local
            pair_norm = pair
        else:
            local_norm = hk.LayerNorm([-1], False, False)(drop(local, p=0.25, is_training=not c.eval))
            pair_norm = hk.LayerNorm([-1], False, False)(drop(pair, p=0.25, is_training=not c.eval))
        if c.potts == "linear":
            pssm_term = MLP(
                local.shape[-1] * 2, 20,
                activation=jax.nn.silu, bias=False, final_init="zeros")(
                    local_norm)
            contact_term = MLP(
                pair.shape[-1] * 2, 20 * 20,
                activation=jax.nn.silu, bias=False, final_init="zeros")(
                    pair_norm).reshape(*pair.shape[:2], 20, 20)
            non_self_mask = (neighbours != jnp.arange(neighbours.shape[0])[:, None]) * (neighbours != -1)
            contact_term = contact_term * non_self_mask[..., None, None]
            pssm_term += hk.get_parameter("aa_bias", (20,), init=init_zeros())
        elif c.potts == "caliby":
            scale = jnp.exp(hk.get_parameter("E_scale", (), init=hk.initializers.Constant(jnp.log(0.1))))
            pssm_term = scale * Linear(20, bias=False, initializer="linear")(local_norm)
            contact_left = scale * Linear(20 * 20, initializer="linear")(
                pair_norm).reshape(*pair.shape[:2], 20, 20)
            contact_right = scale * Linear(20 * 20, initializer="linear")(
                pair_norm).reshape(*pair.shape[:2], 20, 20)
            contact = jnp.einsum("...ac,...cb->...ab", contact_left, contact_right)
            contact = drop(contact, is_training=not c.eval)
            contact_term = contact
            pssm_term -= pssm_term.mean(axis=-1, keepdims=True)
            contact = (
                contact
                - contact.mean(axis=-1, keepdims=True)
                - contact.mean(axis=-2, keepdims=True)
                + contact.mean(axis=(-1, -2), keepdims=True)
            )
            non_self_mask = (neighbours != jnp.arange(neighbours.shape[0])[:, None]) * (neighbours != -1)
            contact_term = contact_term * non_self_mask[..., None, None]
            pssm_term += hk.get_parameter("aa_bias", (20,), init=init_zeros())
        result = dict(local=local, pair=pair, extra_pair=extra_pair,
                      pssm_term=pssm_term, contact_term=contact_term,
                      neighbours=neighbours)
        # compute pseudo-likelihood
        aa_gt = jax.nn.one_hot(data["aa_gt"], 20)

        # construct field-only prediction
        result["aa"] = jax.nn.log_softmax(pssm_term, axis=-1)
        # construct standard pseudo-likelihood (full sequence w/o one AA)
        result["aa_pseudo"] = _pseudo_likelihood(
            pssm_term, contact_term,
            aa_gt[:, None], aa_gt[neighbours],
            neighbours)
        result["aa_pseudo_log_p"], _ = _pair_pseudo_likelihood(
            pssm_term, contact_term, aa_gt, mask, neighbours, smoothing=0.1)
        # _pseudo_likelihood(
        #     pssm_term, contact_term,
        #     aa_gt[:, None], aa_gt[neighbours],
        #     neighbours)
        # construct autoregressive pseudo-likelihood
        order = jax.random.permutation(hk.next_rng_key(), aa_gt.shape[0])
        autoregressive_mask = order[:, None] > order[neighbours]
        autoregressive_neighbours = jnp.where(autoregressive_mask, neighbours, -1)
        autoregressive_contact_term = jnp.where(
            autoregressive_mask[..., None, None], contact_term, 0.0)
        # pssm_softmax = jax.nn.softmax(result["aa"])
        result["aa_ar"] = _pseudo_likelihood(
            pssm_term,
            autoregressive_contact_term,
            jnp.where(autoregressive_mask[..., None], 0.0, aa_gt[:, None]),
            jnp.where(autoregressive_mask[..., None], aa_gt[neighbours], 0.0),
            neighbours)
        result["aa_ar_log_p"], _ = _pair_pseudo_likelihood(
            pssm_term, autoregressive_contact_term, aa_gt, mask,
            autoregressive_neighbours, smoothing=0.1)
        # construct self-consistent pseudo-likelihood
        pssm = jax.nn.softmax(10 * result["aa"])
        for i in range(2):
            pssm = jax.nn.softmax(10 * _pseudo_likelihood(
                pssm_term, contact_term,
                pssm[:, None], pssm[neighbours],
                neighbours))
        pssm = jax.lax.stop_gradient(pssm)
        result["aa_sc"] = _pseudo_likelihood(
            pssm_term, contact_term,
            pssm[:, None], pssm[neighbours],
            neighbours)
        # compute loss
        total, losses = self.loss(result, data)
        result["losses"] = losses
        return total, result

    def prepare_data(self, data):
        c = self.config
        batch = data["batch_index"]
        pos = data["all_atom_positions"]
        print("POS_SHAPE", pos.shape)
        smol_pos = data["smol_positions"]
        smol_mask = data["smol_mask"]
        drop_smol = jax.random.bernoulli(hk.next_rng_key(), 0.5, batch.shape)[batch]
        smol_mask = smol_mask * (1 - drop_smol)[:, None] > 0
        if c.pos_noise and not c.eval:
            pos += c.pos_noise * jax.random.normal(hk.next_rng_key(), pos.shape)
            # smol_pos += c.pos_noise * jax.random.normal(hk.next_rng_key(), smol_pos.shape)
        pos, smol_pos = augment_rotation(hk.next_rng_key(), [pos, smol_pos])
        data["all_atom_position"] = pos
        data["smol_positions"] = smol_pos
        print("POS_SHAPE", pos.shape)
        return data

    def loss(self, result: dict, data: dict):
        c = self.config
        losses = dict()
        total = 0.0
        # negative log likelihood on PSSM features
        aa_mask = (data["aa_gt"] < 20) * data["mask"]
        neighbour_mask = (result["neighbours"] != -1) * (data["mask"][result["neighbours"]])
        for name, item in result.items():
            if name.endswith("log_p"):
                per_residue_loss = item
                losses[name] = -per_residue_loss.sum() / jnp.maximum(aa_mask.sum(), 1)
                total += c.losses[name] * losses[name]
            elif name.startswith("aa"):
                aa = item
                target = jax.nn.one_hot(data["aa_gt"], 20)
                if c.label_smoothing:
                    alpha = c.label_smoothing
                    uniform = jnp.ones_like(target) / 20
                    target = (1 - alpha) * target + alpha * uniform
                aa_nll = -(target * aa).sum(axis=-1)
                aa_nll = (aa_nll * aa_mask).sum() / jnp.maximum(1, aa_mask.sum())
                losses[name] = aa_nll
                total += c.losses[name] * aa_nll
        # regularization
        losses["pssm_term"] = ((result["pssm_term"] ** 2).sum(axis=-1) * aa_mask).sum() / jnp.maximum(1, aa_mask.sum())
        total += c.losses["pssm_term"] * losses["pssm_term"]
        losses["contact_term"] = ((result["contact_term"] ** 2).sum(axis=(-1, -2)) * neighbour_mask).sum() / jnp.maximum(1, neighbour_mask.sum())
        total += c.losses["contact_term"] * losses["contact_term"]
        return total, losses

    def prepare_features(self, data):
        c = self.config
        pos = data["all_atom_positions"]
        atom_mask = data["all_atom_mask"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = atom_mask.any(axis=1)
        same_batch = batch[:, None] == batch[None, :]
        pair_mask = same_batch * (mask[:, None] * mask[None, :])
        pos = positions_to_ncacocb(pos)
        pos = Vec3Array.from_array(pos)
        #centers = pos[:, 1]
        #centers = Vec3Array.from_array(centers)
        delta = pos[:, None, 1] - pos[None, :, 1]
        dist = delta.norm()
        neighbours = get_neighbours(c.num_neighbours)(dist, pair_mask)
        delta = pos[:, None, :, None] - pos[neighbours, None, :]
        print("DELTA SHAPE", delta.shape)
        dist = delta.norm()
        dirs = delta.normalized().to_array()
        print("DIST SHAPE", dist.shape, "DIRS SHAPE", dirs.shape)

        # TODO: move everything over to N-by-N distances
        # TODO: refactor this into a separate function
        smol_atom_pos = data["smol_positions"]
        smol_atom_type = data["smol_types"]
        smol_atom_mask = data["smol_mask"]
        # randomly drop small molecule neighbours
        if not c.eval:
            smol_atom_mask *= jax.random.bernoulli(hk.next_rng_key(), 0.1, smol_atom_mask.shape)
        # FIXME: put a cap on small molecule neighbours
        atom_dist = (pos[:, None, 1] - Vec3Array.from_array(smol_atom_pos)).norm()
        smol_atom_mask = smol_atom_mask * (atom_dist < 10.0)
        # TODO: optionally produce "backbone atoms" for small molecules
        smol_atom_environment = jnp.zeros(list(smol_atom_pos.shape[:2]) + [5, 3], dtype=jnp.float32)
        if c.learn_smol_env:
            smol_atom_environment = SMOLEnv(c)(
                smol_atom_pos, smol_atom_type, smol_atom_mask)
        smol_atom_pos = smol_atom_pos[:, :, None, :] + smol_atom_environment
        # TODO: maybe have neighbours also have neighbours?
        atom_delta = pos[:, None, :, None] - Vec3Array.from_array(smol_atom_pos[:, :, None, :])
        atom_dist = atom_delta.norm()
        atom_dirs = atom_delta.normalized().to_array()
        all_dist = jnp.concatenate((dist, atom_dist), axis=1)
        all_dirs = jnp.concatenate((dirs, atom_dirs), axis=1)
        all_atom_type = jnp.concatenate((jnp.zeros_like(neighbours), smol_atom_type + 1), axis=1)

        # prepare pair features
        environment_embedding = None
        if c.environment == "orb":
            environment_embedding = orb_environment(c.num_rbf, c.l_max, center_only=c.ca_only)
        elif c.environment == "additive_orb":
            environment_embedding = additive_orb_environment(c.num_rbf, c.l_max, center_only=c.ca_only)
        else:
            environment_embedding = dist_environment(c.num_rbf, center_only=c.ca_only)
        environment = environment_embedding(all_dist, all_dirs)
        pair_features = [
            environment,
            jax.nn.one_hot(all_atom_type, 109),
        ]
        pair_mask = mask[:, None] * (neighbours != -1) > 0
        extra_pair_mask = mask[:, None] * smol_atom_mask
        full_pair_mask = jnp.concatenate((pair_mask, extra_pair_mask), axis=1)
        if c.resi_pair_features:
            residue_distance = jnp.clip(resi[:, None] - resi[neighbours], -32, 32) + 32
            residue_distance = jnp.where(chain[:, None] != chain[neighbours], 65, residue_distance)
            residue_distance = jnp.concatenate((residue_distance, jnp.full_like(extra_pair_mask, 66, dtype=jnp.int32)), axis=1)
            residue_distance = jax.nn.one_hot(residue_distance, 66)
            pair_features.append(residue_distance)
        pair_features = jnp.concatenate(pair_features, axis=-1)
        all_pair = MLP(c.pair_size * 2, c.pair_size, activation=jax.nn.silu, final_init="linear")(pair_features)
        all_pair = hk.LayerNorm([-1], False, False)(all_pair)
        pair = all_pair[:, :neighbours.shape[1]]
        extra_pair = all_pair[:, neighbours.shape[1]:]
        extra_pos = atom_delta.to_array()[:, :, 1]

        # prepare local features
        pair_update = Linear(c.local_size, bias=False)(pair)
        local = 0.0
        if c.local_env:
            contacts = all_dist < 6.0
            contacts = full_pair_mask * contacts[:, :, 0, 0] > 0
            local_environment = jnp.where(contacts[..., None], environment, 0).sum(axis=1)
            local += Linear(c.local_size, initializer="linear", bias=False)(local_environment)
        local += jnp.where((neighbours != -1)[..., None], drop(pair_update, is_training=not c.eval), 0).sum(axis=1)
        local = hk.LayerNorm([-1], False, False)(local)

        return local, pair, pos.to_array(), extra_pair, extra_pair_mask, extra_pos, neighbours, mask

class SequenceModel(PottsSequenceModel):
    def __call__(self, data):
        c = self.config
        # prepare data
        data.update(self.prepare_data(data))
        # data.update(self.apply_diffusion(data))
        # run model
        local, pair, pos, extra_pair, extra_pair_mask, extra_pos, neighbours, mask = self.prepare_features(data)
        mpnn_block = None
        if c.block_type == "extended":
            mpnn_block = ExtendedBlock
        else:
            mpnn_block = MPNNBlock
        def body(carry):
            local, pair, extra_pair = carry
            local, pair, extra_pair = mpnn_block(c)(
                local, pair, pos, extra_pos, extra_pair,
                extra_pair_mask, neighbours, mask)
            return local, pair, extra_pair
        local, pair, extra_pair = layer_stack(c.depth, with_state=False)(hk.remat(body))((local, pair, extra_pair))
        if not c.no_renorm:
            local = hk.LayerNorm([-1], False, False)(drop(local, p=0.25, is_training=not c.eval))
            pair = hk.LayerNorm([-1], False, False)(drop(pair, p=0.25, is_training=not c.eval))
        decoder_results = dict()
        losses = dict()
        total = 0.0
        for decoder in c.decoders:
            if decoder == "potts":
                model = PottsDecoder(c)
                decoder_results[decoder] = model(
                    local, pair, extra_pair, neighbours, extra_pair_mask, mask)
                losses[decoder] = model.loss(decoder_results[decoder], data)
            elif decoder == "adm":
                p_mask = jax.random.uniform(hk.next_rng_key(), local.shape[0])[data["batch_index"]]
                aa_masked = jnp.where(jax.random.bernoulli(hk.next_rng_key(), p_mask), data["aa_gt"], 20)
                model = ADMDecoder(c)
                decoder_results[decoder] = model(
                    aa_masked, local, pair, pos, extra_pos,
                    extra_pair, neighbours, extra_pair_mask, mask)
                losses[decoder] = model.loss(decoder_results[decoder], data)
            else:
                raise NotImplementedError("Currently only Potts and ADM Decoders are implemented.")
            for name, weight in c.decoders[decoder].items():
                if name in losses[decoder]:
                    total += weight * losses[decoder][name]
        result = dict(results=decoder_results, losses=flatten_dict(losses))
        return total, result

def flatten_dict(data):
    result = dict()
    for key, item in data.items():
        if isinstance(item, dict):
            update = {key + "_" + k: v for k, v in flatten_dict(item).items()}
            result.update(update)
        else:
            result[key] = item
    return result

class PottsDecoder(hk.Module):
    def __init__(self, config, name = "potts_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, extra_pair, neighbours, extra_pair_mask, mask):
        c = self.config
        if c.potts == "linear":
            pssm_term = MLP(
                local.shape[-1] * 2, 20,
                activation=jax.nn.silu, bias=False, final_init="zeros")(
                    local)
            contact_term = MLP(
                pair.shape[-1] * 2, 20 * 20,
                activation=jax.nn.silu, bias=False, final_init="zeros")(
                    pair).reshape(*pair.shape[:2], 20, 20)
            non_self_mask = (neighbours != jnp.arange(neighbours.shape[0])[:, None]) * (neighbours != -1)
            contact_term = contact_term * non_self_mask[..., None, None]
            pssm_term += hk.get_parameter("aa_bias", (20,), init=init_zeros())
        elif c.potts == "caliby":
            scale = jnp.exp(hk.get_parameter("E_scale", (), init=hk.initializers.Constant(jnp.log(0.1))))
            pssm_term = scale * Linear(20, bias=False, initializer="linear")(local)
            contact_left = scale * Linear(20 * 20, initializer="linear")(
                pair).reshape(*pair.shape[:2], 20, 20)
            contact_right = scale * Linear(20 * 20, initializer="linear")(
                pair).reshape(*pair.shape[:2], 20, 20)
            contact = jnp.einsum("...ac,...cb->...ab", contact_left, contact_right)
            contact = drop(contact, is_training=not c.eval)
            contact_term = contact
            pssm_term -= pssm_term.mean(axis=-1, keepdims=True)
            contact = (
                contact
                - contact.mean(axis=-1, keepdims=True)
                - contact.mean(axis=-2, keepdims=True)
                + contact.mean(axis=(-1, -2), keepdims=True)
            )
            non_self_mask = (neighbours != jnp.arange(neighbours.shape[0])[:, None]) * (neighbours != -1)
            contact_term = contact_term * non_self_mask[..., None, None]
        return dict(pssm_term=pssm_term, contact_term=contact_term, neighbours=neighbours)

    def loss(self, result, data):
        pssm_term = result["pssm_term"]
        contact_term = result["contact_term"]
        neighbours = result["neighbours"]
        aa_gt = data["aa_gt"]
        mask = data["mask"] * (aa_gt != 20)
        aa_gt = jax.nn.one_hot(data["aa_gt"], 20)
        losses = dict()
        losses["aa"] = -(
            mask * (jax.nn.log_softmax(pssm_term, axis=-1)
                    * aa_gt).sum(axis=-1)).sum() / jnp.maximum(mask.sum(), 1)
        losses["aa_pseudo_log_p"] = -_pair_pseudo_likelihood(
            pssm_term, contact_term, aa_gt, mask,
            neighbours, smoothing=0.1)[0].sum() / jnp.maximum(mask.sum(), 1)
        order = jax.random.permutation(hk.next_rng_key(), aa_gt.shape[0])
        autoregressive_mask = order[:, None] > order[neighbours]
        autoregressive_neighbours = jnp.where(autoregressive_mask, neighbours, -1)
        autoregressive_contact_term = jnp.where(
            autoregressive_mask[..., None, None], contact_term, 0.0)
        losses["aa_ar_log_p"] = -_pair_pseudo_likelihood(
            pssm_term, autoregressive_contact_term, aa_gt, mask,
            autoregressive_neighbours, smoothing=0.1)[0].sum() / jnp.maximum(mask.sum(), 1)
        return losses
        
class ADMDecoder(hk.Module):
    def __init__(self, config, name = "adm_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa_masked, local, pair, pos,
                 extra_pos, extra_pair, neighbours,
                 extra_pair_mask, mask):
        c = self.config
        aa_one_hot = jax.nn.one_hot(aa_masked, 21, axis=-1)
        local += Linear(local.shape[-1], bias=False, initializer="linear")(
            aa_one_hot)
        local = hk.LayerNorm([-1], True, True)(local)
        pair += Linear(pair.shape[-1], bias=False, initializer="linear")((
            aa_one_hot[:, None, :, None] *
            aa_one_hot[neighbours, None, :]).reshape(*neighbours.shape, -1))
        
        def body(carry):
            local, pair, extra_pair = carry
            local, pair, extra_pair = ExtendedBlock(c)(
                local, pair, pos,
                extra_pos, extra_pair, extra_pair_mask,
                neighbours, mask)
            return local, pair, extra_pair

        local, *_ = layer_stack(c.decoder_depth, with_state=False)(hk.remat(body))((local, pair, extra_pair))
        aa = jax.nn.log_softmax(Linear(20, bias=True, initializer="zeros")(local))
        result = dict(aa=aa, aa_masked=aa_masked)
        return result

    def loss(self, result, data):
        aa_gt = data["aa_gt"]
        mask = data["mask"] * (aa_gt != 20) * (result["aa_masked"] == 20)
        losses = dict()
        losses["aa"] = -(
            mask * (jax.nn.log_softmax(result["aa"], axis=-1)
                    * jax.nn.one_hot(aa_gt, 20)).sum(axis=-1)).sum() / jnp.maximum(mask.sum(), 1)
        return losses

class SMOLEnv(hk.Module):
    def __init__(self, config, name = "smol_env"):
        super().__init__(name)
        self.config = config

    def __call__(self, smol_atom_pos, smol_atom_type, smol_atom_mask):
        pair_mask = smol_atom_mask[:, :, None] * smol_atom_mask[:, None, :] > 0
        pos = Vec3Array.from_array(smol_atom_pos)
        delta = (pos[:, :, None] - pos[:, None, :])
        dist = delta.norm()
        dist = gaussian_rbf(dist, *linear_centers(0.0, 10.0, bins=16))
        weight = Linear(5, bias=False)(dist)
        weight = jnp.where(pair_mask[..., None], weight, -1e9)
        weight = jax.nn.softmax(weight, axis=2)
        weight = jnp.where(pair_mask[..., None], weight, 0.0)
        result = jnp.einsum("inmh,imd->inhd", weight, smol_atom_pos)
        result = result.at[:, :, 1, :].set(0.0)
        print("SMOL_ENV", result.shape)
        return result

def orb_environment(bins=64, l_max=3, center_only=True):
    def _env(dist, dirs):
        if center_only:
            dist = dist[:, :, 1, 1]
            dirs = dirs[:, :, 1, 1]
        rbfs = gaussian_rbf(dist, *log_centers(0.0, 20.0, bins=bins))
        print("RBF", rbfs.shape)
        spharm = spherical_harmonics(dirs, l_max=l_max)
        print("SPHARM", spharm.shape)
        result = (rbfs[..., :, None] * spharm[..., None, :]).reshape(*dist.shape[:2], -1)
        print("ENV", result.shape)
        return result
    return _env

def additive_orb_environment(bins=64, l_max=3, center_only=True):
    def _env(dist, dirs):
        if center_only:
            dist = dist[:, :, 1, 1]
            dirs = dirs[:, :, 1, 1]
        rbfs = gaussian_rbf(dist, *log_centers(0.0, 20.0, bins=bins))
        spharm = spherical_harmonics(dirs, l_max=l_max)
        result = jnp.concatenate((rbfs, spharm), axis=-1).reshape(*dist.shape[:2], -1)
        print("ENV", result.shape)
        return result
    return _env

def dist_environment(bins=64, center_only=True):
    def _env(dist, dirs):
        if center_only:
            dist = dist[:, :, 1, 1]
            dirs = dirs[:, :, 1, 1]
        rbfs = gaussian_rbf(dist, *linear_centers(0.0, 20.0, bins=bins))
        return rbfs.reshape(*dist.shape[:2], -1)
    return _env

def _pseudo_likelihood(h_i, J_ij, aa_outgoing, aa_incoming, neighbours):
    incoming = jnp.einsum("inab,ina->ib", J_ij, aa_incoming)
    outgoing = jnp.zeros_like(incoming).at[neighbours].add(
        jnp.einsum("inab,inb->ina", J_ij, aa_outgoing))
    scores = incoming + outgoing + h_i
    result = jax.nn.log_softmax(scores, axis=-1)
    return result

def _pair_pseudo_likelihood(h_i, J_ij_ab, aa_i, mask, neighbours, smoothing=0.0):
    pair_mask = mask[:, None] * mask[neighbours] * (neighbours != -1)
    h_i = jnp.where(mask[:, None], h_i, 0.0)
    J_ij_ab = jnp.where(pair_mask[..., None, None], J_ij_ab, 0.0)
    aa_j = aa_i[neighbours]
    J_ij_a = jnp.einsum("ijab,ijb->ija", J_ij_ab, aa_j)
    J_ij_b = jnp.einsum("ijab,ia->ijb", J_ij_ab, aa_i)
    r_i = h_i + J_ij_a.sum(axis=1)
    r_j = r_i[neighbours]
    r_i_minus_ij = r_i[:, None, :, None] - J_ij_a[:, :, :, None]
    r_j_minus_ji = r_j[:, :, None, :] - J_ij_b[:, :, :, None]
    score_ij = r_i_minus_ij + r_j_minus_ji + J_ij_ab
    score_ij = jax.nn.log_softmax(score_ij, axis=(-1, -2))
    log_p_j = jnp.einsum("ijab,ijb->ija", score_ij, aa_j)
    log_p_ij = jnp.einsum("ija,ia->ij", log_p_j, aa_i)
    if smoothing > 0.0:
        N = h_i.shape[-1] ** 2
        alpha = smoothing
        p_no_smooth = (1 - alpha) ** 2
        p_bg = (1 - p_no_smooth) / (N - 1)
        p_fg = p_no_smooth - p_bg
        log_p_ij = p_fg * log_p_ij + p_bg * score_ij.sum(axis=(-1, -2))
    log_p_ij = jnp.where(pair_mask, log_p_ij, 0.0)
    log_p_i = mask * log_p_ij.sum(axis=-1) / jnp.maximum(2 * pair_mask.sum(axis=-1), 2)
    return log_p_i, log_p_ij

class OrbMessage(hk.Module):
    def __init__(self, size, direction="incoming",
                 r_max=None, p=3.0, name = "orb_message"):
        super().__init__(name)
        self.size = size
        self.direction = direction
        self.r_max = r_max
        self.p = p

    def __call__(self, pair, pair_update, neighbours, pair_mask, distance=None):
        message = jax.nn.sigmoid(Linear(self.size, bias=False)(pair)) * pair_update
        message = jnp.where(pair_mask[..., None], message, 0)
        if self.r_max is not None and distance is not None:
            radius_cutoff = _envelope(distance, self.r_max, self.p)
            message *= radius_cutoff[:, None, None]
        if self.direction == "incoming":
            return message.sum(axis=1)
        elif self.direction == "outgoing":
            return jnp.zeros((pair.shape[0], self.size), dtype=pair.dtype).at[neighbours].add(message)
        else:
            raise NotImplementedError(
                f"Unknown direction: {self.direction}. Use either 'incoming' or 'outgoing'.")

# TODO:
# def expand_potts(aa, h_i, J_in, neighbours):
#     index = jnp.arange(h_i.shape[0], dtype=jnp.int32)
#     J_full = jnp.zeros((h_i.shape[0], h_i.shape[1], 20, 20), dtype=jnp.float32)
#     J_full = J_full.at[index[:, None], neighbours].add(J_in)
#     J_full = J_full.at[neighbours, index[None, :]].add(J_in)
#     # drop diagonal
#     J_full = J_full.at[index, index].set(0.0)
#     # pseudo
#     log_aa = jax.nn.log_softmax(jnp.einsum("ijab,jb->ia", J_full, aa) + h_i[:, None] + h_i[None, :], axis=-1)
#     log_aa_pair = jax.nn.log_softmax(jnp.einsum(""))

def _envelope(distance, r_max=6.0, p=4):
    # Adapted from the ORB codebase
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * (distance / r_max) ** p
        + p * (p + 2.0) * (distance / r_max) ** (p + 1)
        - (p * (p + 1.0) / 2) * (distance / r_max) ** (p + 2)
    )
    cutoff = (envelope * (distance < r_max))
    return cutoff

class OrbBlock(hk.Module):
    def __init__(self, config, name = "orb_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, neighbours, mask):
        # This is pretty much the Orb architecture
        # FIXME: how do we integrate small molecules?
        # this could be either be as additional pairs, or as full atoms
        pair_mask = (neighbours != -1) * mask[neighbours] > 0
        pair_features = jnp.concatenate((
            pair,
            jnp.broadcast_to(local[:, None, :], pair.shape),
            local[neighbours, :],
        ), axis=-1)
        pair_update = GatedMLP(pair.shape[-1] * 2, pair.shape[-1], final_init="linear")(pair_features)
        pair_update = hk.LayerNorm([-1], True, True)(pair_update)
        # FIXME: do we need radius cutoff?
        incoming_message = jax.nn.sigmoid(Linear(local.shape[-1], bias=False)(pair)) * pair_update
        incoming_message = jnp.where(pair_mask[..., None], incoming_message, 0)
        incoming_message = incoming_message.sum(axis=1)
        outgoing_message = jax.nn.sigmoid(Linear(local.shape[-1], bias=False)(pair)) * pair_update
        outgoing_message = jnp.where(pair_mask[..., None], outgoing_message, 0)
        outgoing_message = jnp.zeros_like(local).at[neighbours].add(outgoing_message)
        local_features = jnp.concatenate((local, incoming_message, outgoing_message), axis=-1)
        local_update = GatedMLP(local.shape[-1] * 2, local.shape[-1], final_init="linear")(local_features)
        local_update = hk.LayerNorm([-1], True, True)(local_update)
        local += local_update
        pair += pair_update
        return local, pair

class MPNNBlock(hk.Module):
    def __init__(self, config, name = "mpnn_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, pos, extra_pos, extra_pair, extra_pair_mask, neighbours, mask):
        c = self.config
        pair_mask = (neighbours != -1) * mask[neighbours] > 0
        full_pair_mask = jnp.concatenate((
            pair_mask, extra_pair_mask), axis=1)
        full_pair = jnp.concatenate((pair, extra_pair), axis=1)
        full_neighbours = jnp.concatenate((local[neighbours], Linear(local.shape[-1], bias=False)(extra_pair)), axis=1)
        full_center = jnp.broadcast_to(local[:, None, :], full_pair.shape)

        # pair update
        pair_features = [full_pair, full_center, full_neighbours]
        pair_features = jnp.concatenate(pair_features, axis=-1)
        full_pair_update = GatedMLP(pair.shape[-1] * c.factor, pair.shape[-1], final_init="linear")(pair_features)
        full_pair_update = drop(full_pair_update, is_training=not c.eval)
        full_pair = hk.LayerNorm([-1], True, True)(full_pair + full_pair_update)
        full_pair_update = hk.LayerNorm([-1], True, True)(full_pair_update)
        pair = full_pair[:, :neighbours.shape[1]]
        extra_pair = full_pair[:, neighbours.shape[1]:]
        # local message
        full_pair = jnp.concatenate((pair, extra_pair), axis=1)
        full_neighbours = jnp.concatenate((local[neighbours], Linear(local.shape[-1], bias=False)(extra_pair)), axis=1)
        full_center = jnp.broadcast_to(local[:, None, :], full_pair.shape)
        pair_features = [full_pair, full_center, full_neighbours]
        pair_features = jnp.concatenate(pair_features, axis=-1)
        message = GatedMLP(pair.shape[-1] * c.factor, local.shape[-1], final_init="linear")(pair_features)
        message = jnp.where(full_pair_mask[..., None], message, 0).sum(axis=1) / full_pair.shape[1]
        local = hk.LayerNorm([-1], True, True)(local + drop(message, is_training=not c.eval))
        # local update
        local_update = GatedMLP(local.shape[-1] * c.factor, local.shape[-1], final_init="linear")(local)
        local = hk.LayerNorm([-1], True, True)(local + drop(local_update, is_training=not c.eval))
        return local, pair, extra_pair

# TODO: make decoder version of this
class ExtendedBlock(hk.Module):
    def __init__(self, config, name = "extended_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, pos, extra_pos, extra_pair, extra_pair_mask, neighbours, mask):
        c = self.config
        print(local.shape, pair.shape, pos.shape, extra_pos.shape, extra_pair.shape, extra_pair_mask.shape, neighbours.shape, mask.shape)
        pair_mask = (neighbours != -1) * mask[neighbours] > 0
        full_pair_mask = jnp.concatenate((
            pair_mask, extra_pair_mask), axis=1)
        # pair attention
        full_pair = jnp.concatenate((pair, extra_pair), axis=1)
        full_neighbours = jnp.concatenate((local[neighbours], Linear(local.shape[-1], bias=False)(extra_pair)), axis=1)
        full_center = jnp.broadcast_to(local[:, None, :], full_pair.shape)
        if c.pair_attention:
            full_pair_attn_update = drop(PairAttention(c)(full_pair, full_pair_mask), is_training=not c.eval)
            if c.norm == "orb":
                full_pair += full_pair_attn_update
            elif c.norm == "post":
                full_pair = hk.LayerNorm([-1], True, True)(full_pair + full_pair_attn_update)
        # regular pair update
        pair_features = [full_pair, full_center, full_neighbours]
        # FIXME: optionally include additional position features
        if c.embed_pos: # TODO
            if not c.eval:
                pos += 0.5 * jax.random.normal(hk.next_rng_key(), pos.shape)
                extra_pos += 0.5 * jax.random.normal(hk.next_rng_key(), extra_pos.shape)
            query_pos = Vec3Array.from_array(
                pos[:, None, :] + Linear(c.key_points * 3, bias=False)(local).reshape(local.shape[0], -1, 3))
            key_pos = Vec3Array.from_array(
                pos[:, None, :] + Linear(c.key_points * 3, bias=False)(local).reshape(local.shape[0], -1, 3))
            extra_key_pos = Vec3Array.from_array(
                extra_pos[:, :, None, :] + Linear(c.key_points * 3, bias=False)(extra_pair).reshape(*extra_pair.shape[:2], -1, 3))
            deltas = (key_pos[neighbours, None, :] - query_pos[:, None, :, None]).to_array()
            extra_deltas = (extra_key_pos[:, :, None, :] - query_pos[:, None, :, None]).to_array()
            full_deltas = jnp.concatenate((deltas, extra_deltas), axis=1)
            features = freq_embedding(full_deltas, 16, 1e-3, 2 ** 8).reshape(*full_deltas.shape[:2], -1)
            pair_features.append(features)
        pair_features = jnp.concatenate(pair_features, axis=-1)
        # FIXME: c.factor 2 ?
        full_pair_update = GatedMLP(pair.shape[-1] * c.factor, pair.shape[-1], final_init="linear")(pair_features)
        if c.norm == "orb":
            full_pair_update = hk.LayerNorm([-1], True, True)(full_pair_update)
            full_pair += drop(full_pair_update, is_training=not c.eval)
        elif c.norm == "post":
            full_pair_update = drop(full_pair_update, is_training=not c.eval)
            full_pair = hk.LayerNorm([-1], True, True)(full_pair + full_pair_update)
            full_pair_update = hk.LayerNorm([-1], True, True)(full_pair_update)
        pair_update = full_pair_update[:, :neighbours.shape[1]]
        pair = full_pair[:, :neighbours.shape[1]]
        extra_pair = full_pair[:, neighbours.shape[1]:]
        # local update
        local_features = jnp.concatenate((
            local,
            drop(OrbMessage(local.shape[-1], direction="incoming")(
                 full_pair, full_pair_update, None, full_pair_mask), is_training=not c.eval),
            drop(OrbMessage(local.shape[-1], direction="outgoing")(
                 pair, pair_update, neighbours, pair_mask), is_training=not c.eval),
        ), axis=-1)
        # FIXME: c.factor 2 ?
        local_update = GatedMLP(local.shape[-1] * c.factor, local.shape[-1], final_init="linear")(local_features)
        if c.norm == "orb":
            local_update = hk.LayerNorm([-1], True, True)(local_update)
            local += drop(local_update, is_training=not c.eval)
        elif c.norm == "post":
            local = hk.LayerNorm([-1], True, True)(local + drop(local_update, is_training=not c.eval))
        return local, pair, extra_pair

class PairAttention(hk.Module):
    def __init__(self, config, name = "pair_attention"):
        super().__init__(name)
        self.config = config

    def __call__(self, pair, pair_mask):
        c = self.config
        def attn_component(x):
            result = Linear(c.key_size * c.heads, bias=False, initializer="glorot")(x)
            return result.reshape(*x.shape[:-1], c.heads, c.key_size)
        pattn_mask = pair_mask[:, None, :, None] * pair_mask[:, None,  None, :]
        query = hk.LayerNorm([-1], False, False)(attn_component(pair))
        key = hk.LayerNorm([-1], False, False)(attn_component(pair))
        value = attn_component(pair)
        result = jax.nn.dot_product_attention(query, key, value, mask=pattn_mask)
        return hk.LayerNorm([-1], True, True)(
            Linear(initializer="linear")(result.reshape(*pair.shape[:2], -1)))
