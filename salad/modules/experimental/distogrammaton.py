from typing import Tuple, Dict

import jax
from jaxtyping import Float, Int, Bool
import jax.numpy as jnp
from salad.aflib.model.layer_stack import layer_stack

import haiku as hk
from salad.modules.basic import Linear, MLP, GatedMLP, init_glorot
from salad.modules.transformer import drop
from salad.modules.geometric import FastTriangle, Triangle

from salad.data.allpdb import RESTYPE_ONEHOT
_RESTYPE_ONEHOT = jnp.array(RESTYPE_ONEHOT, dtype=jnp.bool_)

def _nll_loss(logits, gt, mask, smooth=0.0):
    if smooth > 0:
        gt = jnp.where(gt > 0, gt * (1 - smooth), smooth / (gt.shape[-1] - 1))
    loss = (-(logits * gt).sum(axis=-1) * mask).sum()
    return loss / jnp.maximum(1, mask.sum())

def _unknown(token):
    return (token[:, None] == jnp.array([20, 25, 30, 149], dtype=jnp.int32)).any(axis=-1)

class Distogrammaton(hk.Module):
    def __init__(self, config, name = "distogrammaton"):
        super().__init__(name)
        self.config = config

    def __call__(self, data: Dict[str, jax.Array]) -> Tuple[jax.Array, dict]:
        c = self.config
        num_tokens_out = 20 + 1 + 4 + 1 + 4 + 1
        residue_type = data["residue_type"]
        mask = data["mask"]
        result = dict()
        local, pair, pair_mask = self.prepare_features(data, result)
        result.update(
            init_local=local,
            init_pair=pair,
            pair_mask=pair_mask
        )

        def block(pair):
            return PairBlock(c)(pair, pair_mask)
        pair = layer_stack(c.depth, with_state=False)(hk.remat(block))(pair)
        pair = hk.LayerNorm([-1], True, True)(pair) # we're doing prenorm now
        result["pair"] = pair

        local_contribution = Linear(c.local_size, bias=False)(pair)
        weighting = jax.nn.sigmoid(Linear(c.local_size, bias=False)(pair))
        local = ((local_contribution * weighting) * pair_mask[..., None]).sum(axis=1)
        local = hk.LayerNorm([-1], True, True)(local)
        losses = dict()
        total = 0.0
        # residue prediction
        pair_logits = MLP(c.local_size * 2, num_tokens_out, final_init="zeros")(pair)
        weight_mask = pair_mask
        if not c.eval:
            weight_mask *= jax.random.bernoulli(hk.next_rng_key(), 0.1, weight_mask.shape)
        residue_logit_weights = Linear(1, initializer="linear", bias=False)(pair)[..., 0]
        residue_logit_weights = jnp.where(weight_mask, residue_logit_weights, -1e9)
        residue_logit_weights = jax.nn.softmax(residue_logit_weights)
        residue_logit_weights = jnp.where(weight_mask, residue_logit_weights, 0)
        residue_logits = (pair_logits * residue_logit_weights[..., None] * weight_mask[..., None]).sum(axis=1)
        # residue_logits = residue_logits - (1 - _RESTYPE_ONEHOT[residue_type][:, :31]) * 1e9
        pair_logits = jax.nn.log_softmax(pair_logits, axis=-1)
        residue_logits = jax.nn.log_softmax(residue_logits, axis=-1)
        result["logits"] = residue_logits
        result["pair_logits"] = pair_logits
        token_gt = jax.nn.one_hot(data["aa_gt"], num_tokens_out)
        is_unknown = _unknown(data["aa"])
        losses["aa"] = _nll_loss(
            result["logits"],
            token_gt,
            mask * is_unknown, smooth=0.1)
        losses["pair_aa"] = _nll_loss(
            result["pair_logits"],
            token_gt[:, None],
            pair_mask * is_unknown[:, None], smooth=0.1)
        # dssp prediction
        dssp = MLP(c.local_size * 2, 3, final_init="zeros")(local)
        result["dssp"] = jax.nn.log_softmax(dssp)
        losses["dssp"] = _nll_loss(
            result["dssp"],
            jax.nn.one_hot(data["dssp"], 3),
            mask * (residue_type == 0), smooth=0.1)

        result["losses"] = losses
        result["aa_gt"] = data["aa_gt"]
        result["aa"] = data["aa"]
        total = losses["aa"] + losses["pair_aa"] + losses["dssp"]

        return total, result

    def prepare_features(self, data: Dict[str, jax.Array], result: Dict[str, jax.Array]) -> Tuple[Float, Float]:
        c = self.config
        mask = data["mask"]
        # TODO: no packed batch for distogram model
        #batch = data["batch_index"]
        #pair_mask = batch[:, None] == batch[None, :]
        pair_mask = mask[:, None] * mask[None, :]
        if "distogram" not in data:
            pos = data["distogram_pos"]
            if not c.eval:
                # add noise to positions
                pos += c.noise_level * jax.random.normal(hk.next_rng_key(), pos.shape)
            distogram = self.prepare_distogram(pos)
        else:
            distogram = data["distogram"]
        if "resigram" not in data:
            resigram = self.prepare_resigram(
                data["residue_index"],
                data["chain_index"],
                data["residue_type"],
                pair_mask)
        else:
            resigram = data["resigram"]
        result["distogram"] = distogram
        result["resigram"] = resigram
        result["residue_index"] = data["residue_index"]
        result["chain_index"] = data["chain_index"]
        local_distance_histogram = (distogram * pair_mask[..., None]).sum(axis=1)
        local_distance_histogram = local_distance_histogram / jnp.maximum(
            1e-3, local_distance_histogram.sum(axis=-1, keepdims=True))
        print("shapes", distogram.shape, resigram.shape, local_distance_histogram.shape, data["aa"].shape)
        
        aa_one_hot = jax.nn.one_hot(
            data["aa"],
            20 + 1  # PROTEIN + UNK
            + 4 + 1 # RNA + N
            + 4 + 1 # DNA + DN
            + 118 + 1   # ATOMS
        )
        local = Linear(c.local_size, bias=False, initializer="linear")(
            # jnp.concatenate((
                jax.nn.one_hot(
                    data["aa"],
                    20 + 1  # PROTEIN + UNK
                    + 4 + 1 # RNA + N
                    + 4 + 1 # DNA + DN
                    + 118 + 1   # ATOMS
                ),
                # local_distance_histogram
            # ), axis=-1)
        )
        local = hk.LayerNorm([-1], True, True)(local)
        pair = Linear(c.pair_size, bias=False)(distogram)
        pair += Linear(c.pair_size, bias=False)(resigram)
        pair += Linear(c.pair_size, bias=False)(aa_one_hot)[:, None]
        pair += Linear(c.pair_size, bias=False)(aa_one_hot)[None, :]
        # pair = Linear(c.pair_size, bias=False, initializer="linear")(
        #     jnp.concatenate((
        #         distogram,# resigram,
        #         #Linear(bias=False)(local)[:, None] + Linear(bias=False)(local)[None, :]
        #     ), axis=-1)
        # )
        pair = hk.LayerNorm([-1], True, True)(pair)
        return local, pair, pair_mask

    def prepare_distogram(self, pos: Float) -> Float[jax.Array, "N N"]:
        c = self.config
        # index = jnp.arange(pos.shape[0])
        center = pos # pos[index, distogram_index]
        distance = jnp.linalg.norm(center[:, None] - center[None, :], axis=-1)
        bin_edges = jnp.linspace(c.distogram_min, c.distogram_max, 65)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        return jax.nn.one_hot(jnp.argmin(abs(distance[..., None] - bin_centers), axis=-1), 64)

    def prepare_resigram(self, residue, chain, residue_type, pair_mask: Bool) -> Float[jax.Array, "N N C"]:
        c = self.config
        other_chain = chain[:, None] != chain[None, :]
        other_chain = other_chain * pair_mask
        other_type = residue_type[:, None] != residue_type[None, :]
        other_type = other_type * pair_mask
        smolecule = residue_type > 2
        residue = jnp.where(smolecule, -1, residue)
        signed_residue_distance = residue[:, None] - residue[None, :]
        signed_residue_distance = jnp.clip(
            signed_residue_distance,
            -c.max_resi_distance, c.max_resi_distance)
        signed_residue_distance = jnp.where(
            other_chain + other_type > 0,
            c.max_resi_distance + 1, signed_residue_distance)
        signed_residue_distance = jax.nn.one_hot(
            signed_residue_distance + c.max_resi_distance,
            2 * c.max_resi_distance + 2, axis=-1)
        return jnp.where(pair_mask[..., None], signed_residue_distance, 0)

class PairBlock(hk.Module):
    def __init__(self, config, name = "pair_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, pair, pair_mask):
        c = self.config
        if c.layer_norm == "prenorm":
            pair += drop(FastTriangle(final_init="zeros")(hk.LayerNorm([-1], True, True)(pair), pair_mask),
                        p=c.drop_rate, is_training=not c.eval)
            pair += drop(MLP(pair.shape[-1] * 2, pair.shape[-1], final_init="zeros")(hk.LayerNorm([-1], True, True)(pair)),
                        p=c.drop_rate, is_training=not c.eval)
        elif c.layer_norm == "postnorm":
            pair_update = drop(FastTriangle(final_init="zeros")(pair, pair_mask),
                               p=c.drop_rate, shared_axes=0, is_training=not c.eval)
            pair = hk.LayerNorm([-1], True, True)(pair + pair_update)
            pair_update = drop(MLP(pair.shape[-1] * 2, pair.shape[-1], final_init="zeros")(pair),
                               p=c.drop_rate, shared_axes=None, is_training=not c.eval)
            pair = hk.LayerNorm([-1], True, True)(pair + pair_update)
        #pair = hk.LayerNorm([-1], True, True)(pair + drop(FastTriangle(final_init="zeros")(pair, pair_mask), is_training=False))
        #pair = hk.LayerNorm([-1], True, True)(pair + drop(GatedMLP(
        #    c.pair_size * 2, c.pair_size, activation=jax.nn.silu, final_init="zeros")(pair), is_training=False))
        return pair
