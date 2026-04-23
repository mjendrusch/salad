import jax
import jax.numpy as jnp
import haiku as hk

from salad.modules.utils.geometry import pairwise_distance, get_neighbours, distance_rbf
from salad.modules.basic import Linear, MLP, GatedMLP, init_zeros
from salad.modules.transformer import drop
from salad.aflib.model.layer_stack import layer_stack

# TODO implement dataset with amino acid residues + extra atoms
# implement modular dataset with tasks, atomization, etc

class AllAtomPotts(hk.Module):
    def __init__(self, config, name = "all_atom_potts"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        if not c.eval or "aa" not in data:
            p_mask = 1.0#jax.random.uniform(hk.next_rng_key(), minval=0.5)
            predict_mask = jax.random.bernoulli(
                hk.next_rng_key(), p_mask, data["aa_gt"].shape)
            data["aa"] = jnp.where(predict_mask, 20, data["aa_gt"])
            # if c.do_not_mask:
        predict_mask = jnp.ones_like(data["aa_gt"], dtype=jnp.bool_)
        # embed inputs
        local, pair, neighbours, mask = AtomPairEmbedding(c)(data)
        pair_mask = neighbours != -1
        # message passing
        def body(x):
            local, pair = x
            local, pair = MPNNBlock(c)(local, pair, neighbours, mask)
            return (local, pair)
        local, pair = layer_stack(c.depth, with_state=False)(hk.remat(body))((local, pair))
        # prediction
        aa = jax.nn.log_softmax(Linear(20, bias=False, initializer="zeros")(local))
        aa_pair = jax.nn.log_softmax(
            Linear(20 * 20, bias=False, initializer="zeros")(pair), axis=-1).reshape(
                *neighbours.shape, 20, 20)
        pssm, couplings = self.potts(local, pair, neighbours)
        
        # computes losses
        # option of providing (soft) one-hot to the model for hallucination purposes
        if "aa_one_hot" in data:
            aa_gt = data["aa_one_hot"]
        else:
            aa_gt = jax.nn.one_hot(data["aa_gt"], 20)
        aa_pair_gt = aa_gt[:, None, :, None] * aa_gt[neighbours, None, :]
        aa_nll = -(aa_gt * aa).sum(axis=-1)
        aa_nll = (predict_mask * mask * aa_nll).sum() / jnp.maximum(1, mask.sum())
        aa_pair_nll = -(aa_pair_gt * aa_pair).sum(axis=(-1, -2))
        aa_pair_nll = (predict_mask[:, None] * pair_mask * aa_pair_nll).sum() / jnp.maximum(1, pair_mask.sum())
        log_p_local, log_p_pair, pair_mask = _pair_pseudo_likelihood(
            pssm, couplings, aa_gt, predict_mask * mask, neighbours, c.label_smoothing)
        log_p_local = log_p_local.sum() / jnp.maximum((predict_mask * mask).sum(), 1)
        log_p_pair = log_p_pair.sum() / jnp.maximum(pair_mask.sum(), 1)
        # logits_local, logits_pair = _pseudo_logits(pssm, couplings, aa_gt, mask, neighbours)
        local_nll = -log_p_local
        potts_nll = -log_p_pair
        losses = dict(aa_nll=aa_nll, aa_pair_nll=aa_pair_nll, local_nll=local_nll, potts_nll=potts_nll)
        total = potts_nll + aa_nll + aa_pair_nll
        result = dict(total=total, aa=aa, aa_pair=aa_pair, pssm=pssm, couplings=couplings,
                      neighbours=neighbours, losses=losses, mask=data["mask"])
        return total, result

    def potts(self, local, pair, neighbours):
        c = self.config
        scale = jnp.exp(hk.get_parameter("E_scale", (), init=hk.initializers.Constant(jnp.log(0.1))))
        pssm_term = scale * Linear(20, bias=False, initializer="zeros")(local)
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
        pssm_term += hk.get_parameter("aa_bias", (20,), init=init_zeros())
        return pssm_term, contact_term

    def sample(self, data, T=0.01, Ts=None, num_steps=100, step_size=0.1):
        result = self(data)
        if Ts is None:
            Ts = jnp.full((num_steps,), T, dtype=jnp.float32)
        aa_template = data["aa"]
        aa_mask = aa_template != 20
        aa = jnp.where(aa_mask, aa_template, jax.random.randint(hk.next_rng_key(), aa_mask.shape, 0, 20))
        def proposal(aa):
            energy, logits_i, _ = _pseudo_logits(
                result["pssm"], result["couplings"], aa, data["mask"], result["neighbours"])
            logits_i = jax.nn.log_softmax(logits_i / T, axis=-1)
            balancing = 0.5 * (logits_i - (logits_i * aa).sum(axis=-1))
            rate = jnp.exp(balancing - logits_i)
            transition_logits = logits_i + jnp.log(-jnp.expm1(-step_size * rate))
            p_other_aa = ((1 - aa) * jnp.exp(transition_logits)).sum(axis=-1)
            log_p_same_aa = jnp.log(jnp.maximum(1e-5, 1.0 - p_other_aa))
            log_p_transition = (1 - aa) * transition_logits + aa * log_p_same_aa[..., None]
            return energy, log_p_transition
        def body(carry, T):
            aa = carry
            aa_oh = jax.nn.one_hot(carry, 20)
            energy, transition_logits = proposal(aa_oh)
            aa_new = jnp.where(
                aa_mask, aa_template,
                jax.random.categorical(hk.next_rng_key(), transition_logits))
            aa_new_oh = jax.nn.one_hot(aa_new)
            energy_new, transition_logits_new = proposal(aa_new_oh)
            backward = -energy_new / T + (aa_mask * (transition_logits_new * aa).sum(axis=-1)).sum(axis=-1)
            forward = -energy / T + (aa_mask * (transition_logits * aa_new_oh).sum(axis=-1)).sum(axis=-1)
            accept = jax.random.bernoulli(
                hk.next_rng_key(),
                jnp.minimum(jnp.exp(backward - forward), 1.0))
            aa_new = jnp.where(accept, aa_new, aa)
            return aa_new, T
        aa_final, _ = jax.lax.scan(body, aa, Ts)
        energy, _ = proposal(aa_final)
        return aa_final, energy

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
    score_ij = jax.nn.log_softmax(-score_ij, axis=(-1, -2)) # FIXME: minus-sign
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
    return log_p_i, log_p_ij, pair_mask

def _pseudo_logits(h_i, J_ij_ab, aa_i, mask, neighbours, smoothing=0.0):
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
    score_ij = jax.nn.log_softmax(-score_ij, axis=(-1, -2))
    J_i_a = J_ij_a.sum(axis=1)
    energy_i = h_i + J_i_a
    score_i = jax.nn.log_softmax(-energy_i, axis=1)
    energy = jnp.einsum("ia,ia->", energy_i, aa_i) - 0.5 * jnp.einsum("ia,ia->", J_i_a, aa_i)
    return energy, score_i, score_ij

class AtomPairEmbedding(hk.Module):
    def __init__(self, config, name = "atom_pair_embedding"):
        super().__init__(name=name)
        self.config = config

    def __call__(self, data):
        c = self.config
        # compute neighbour list
        if "is_aa" in data:
            is_aa = data["is_aa"]
        else:
            is_aa = jnp.ones_like(data["aa"], dtype=jnp.bool_)
        if "molecule_index" in data:
            chain_index = data["molecule_index"]
        else:
            chain_index = data["chain_index"]
        residue_index = data["residue_index"]
        positions = data["all_atom_positions"][:, 1]
        if not c.eval:
            positions += c.noise_scale * jax.random.normal(hk.next_rng_key(), positions.shape)
        mask = data["all_atom_mask"][:, 1] > 0
        distance = pairwise_distance(positions, neighbours=None)
        distance = jnp.where(mask[:, None] * mask[None, :], distance, jnp.inf)
        aa_distance = jnp.where(is_aa[None, :], distance, jnp.inf)
        smol_distance = jnp.where(~is_aa[None, :], distance, jnp.inf)
        aa_neighbours = get_neighbours(c.num_aa_neighbours)(
            aa_distance, jnp.isfinite(aa_distance))
        non_aa_neighbours = get_neighbours(c.num_smol_neighbours)(
            smol_distance, jnp.isfinite(smol_distance))
        neighbours = jnp.concatenate((aa_neighbours, non_aa_neighbours), axis=1)
        # set up pair / distance features
        distance = pairwise_distance(positions, neighbours=neighbours)
        distance_features = distance_rbf(distance, min_distance=2.0, max_distance=22.0)
        type_features = is_aa[neighbours][..., None]
        other_chain = (chain_index[:, None] != chain_index[neighbours])[..., None]
        same_residue = (chain_index[:, None] == chain_index[neighbours]) * (residue_index[:, None] == residue_index[neighbours])
        same_residue = same_residue[..., None]
        pair = Linear(c.pair_size, bias=False)(jnp.concatenate((
            distance_features, type_features, same_residue, other_chain), axis=-1))
        pair = hk.LayerNorm([-1], True, True)(pair)
        # set up local features
        type_features = is_aa[..., None]
        aa_features = jax.nn.one_hot(data["aa"], 21, axis=-1)
        pair_mask = neighbours != -1
        pair_weighted = (MLP(c.pair_size * 2, c.local_size)(pair) * pair_mask[..., None]).sum(axis=1)
        local = Linear(c.pair_size, bias=False)(jnp.concatenate((pair_weighted, type_features, aa_features), axis=-1))
        local = hk.LayerNorm([-1], True, True)(local)
        return local, pair, neighbours, mask

class MPNNBlock(hk.Module):
    def __init__(self, config, name = "mpnn_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, neighbours, mask):
        c = self.config
        pair_mask = mask[:, None] * (neighbours != -1)
        # pass messages inward
        local_update = MLP(c.local_size * 4, c.local_size, bias=False)(
            jnp.concatenate((
                jnp.broadcast_to(local[:, None], list(neighbours.shape) + [local.shape[-1]]),
                local[neighbours], pair), axis=-1))
        local_update = (jnp.where(pair_mask[..., None], local_update, 0)).sum(axis=1) / neighbours.shape[1]
        local_update *= jax.nn.sigmoid(Linear(c.local_size, bias=True)(local))
        local_update = drop(local_update, is_training=not c.eval)
        local = hk.LayerNorm([-1], True, True)(local + local_update)
        # update local features
        local_update = GatedMLP(
            local.shape[-1] * 4, local.shape[-1],
            activation=jax.nn.silu, final_init="linear")(local)
        local_update = drop(local_update, is_training=not c.eval)
        local = hk.LayerNorm([-1], True, True)(local + local_update)
        # update pair features
        pair_update = MLP(c.pair_size * 2, c.pair_size, bias=False)(
            jnp.concatenate((
                jnp.broadcast_to(local[:, None], list(neighbours.shape) + [local.shape[-1]]),
                local[neighbours], pair), axis=-1))
        pair_update *= jax.nn.sigmoid(Linear(c.pair_size, bias=True)(pair))
        pair_update = drop(pair_update, is_training=not c.eval)
        pair = hk.LayerNorm([-1], True, True)(pair + pair_update)
        return local, pair
