"""Efficient triangle multiplication ops."""

import jax
import jax.numpy as jnp

import haiku as hk

from salad.aflib.model.geometry import Rigid3Array, Vec3Array

from salad.modules.basic import Linear
from salad.modules.utils.geometry import extract_aa_frames
from salad.modules.transformer import masked_softmax

class FlashPointAttention(hk.Module):
    """Efficient point attention from Liu et al. 2025,
    Flash Invariant Point Attention.
    """
    def __init__(self, key_size=32, heads=8, key_points=8,
                 pair_size=32, pair_rank=2, residue_gamma=False,
                 use_pair=False, equivariant=True, final_init="zeros",
                 backend="xla", key_norm_params=True, overscale_dfactor=True,
                 return_attn=False, point_init="linear", name = "flash_ipa"):
        super().__init__(name=name)
        self.key_size = key_size
        self.heads = heads
        self.key_points = key_points
        self.pair_size = pair_size
        self.pair_rank = pair_rank
        self.residue_gamma = residue_gamma
        self.use_pair = use_pair
        self.equivariant = equivariant
        self.final_init = final_init
        self.backend = backend
        self.key_norm_params = key_norm_params
        self.return_attn = return_attn
        self.overscale_dfactor = overscale_dfactor
        self.point_init = point_init

    def __call__(self, local, frames, mask, pair_bias=None):
        # factorized pair feature computation and reshaping
        def pair_factor(x):
            result = Linear(self.pair_size * self.pair_rank, bias=False)(x)
            return result.reshape(*x.shape[:-1], self.pair_size, self.pair_rank)
        # query / key / value computation and reshaping
        def attention_component(x):
            result = Linear(self.key_size * self.heads, bias=False)(x)
            return result.reshape(*result.shape[:-1], self.heads, self.key_size)
        # query / key / value point computation, projection and reshaping
        def attention_points(frames: Rigid3Array, x):
            result = Linear(3 * self.key_points * self.heads, bias=False, initializer=self.point_init)(x)
            result = result.reshape(*result.shape[:-1], self.heads, self.key_points, 3)
            if self.equivariant:
                result = frames[:, None, None].apply_to_point(
                    Vec3Array.from_array(result))
            else:
                result += frames.translation.to_array()[:, None, None, :]
                result = Vec3Array.from_array(result)
            return result
        frames: Rigid3Array = Rigid3Array.from_array(frames.astype(jnp.float32))
        # compute factored pair features and biases
        if self.use_pair:
            pair_a = pair_factor(local)
            pair_b = pair_factor(local)
            bias_query = Linear(self.heads, bias=False)(
                pair_a.swapaxes(-1, -2)).swapaxes(-1, -2)
            bias_key = Linear(self.heads, bias=False)(
                pair_b.swapaxes(-1, -2)).swapaxes(-1, -2)
        # compute query, key, value features
        use_params = self.key_norm_params
        query = hk.LayerNorm([-1], use_params, use_params)(attention_component(local))
        key = hk.LayerNorm([-1], use_params, use_params)(attention_component(local))
        value = attention_component(local)
        # compute query, key, value points
        query_points = attention_points(frames, local)
        key_points = attention_points(frames, local)
        value_points = attention_points(frames, local)
        # assemble joint query / key
        w_C = jnp.sqrt(2 / (9 * self.key_points))
        w_L = jnp.sqrt(1 / 3)

        # per-head scale factor for point-distance attention
        gamma = hk.get_parameter(
            "gamma", (self.heads,),
            init=hk.initializers.Constant(jnp.log(jnp.exp(1.) - 1.))
        )
        gamma = gamma[None, :, None]
        # optionally have the residue control gamma
        if self.residue_gamma:
            gamma += Linear(self.heads, bias=False, initializer="zeros")(local)[:, None, :]
        dfactor = jax.nn.softplus(gamma)
        if self.overscale_dfactor:
            dfactor *= w_C / 2
        # flatten query, key and value to put them in the correct shape for standard attention
        query_flat = query_points.to_array().reshape(*query.shape[:2], -1)
        query_squared = query_points.norm2()
        key_flat = key_points.to_array().reshape(*key.shape[:2], -1)
        key_squared = key_points.norm2()
        value_flat = value_points.to_array().reshape(*value.shape[:2], -1)

        # combine query and key features in such a way that taking the inner product
        # of query and key results in standard IPA (with factored pair bias)
        # this results in vectors:
        # query: Q / sqrt(dim(Q)) :: 2 * flatten(Q_point) :: -Q_point^2 :: 1 :: bias_q
        # key  : K                ::     flatten(K_point) :: 1 :: -K_point^2 :: bias_k
        # taking the inner product then gives:
        #   Q . K / sqrt(dim(Q))                          # regular attention
        # + 2 * Q_point . K_point - Q_point^2 - K_point^2 # - sum (Q_point - K_point)^2, the point attention term
        # + bias_q . bias_k                               # the (factored) pair bias term
        query_components = [
            query * w_L * jnp.sqrt(1.0 / self.key_size),
            query_flat * (dfactor * w_L * w_C),
            query_squared * (-dfactor * w_L * w_C / 2),
            jnp.ones_like(key_squared) * (-dfactor * w_L * w_C / 2)
        ]
        key_components = [
            key,
            key_flat,
            jnp.ones_like(query_squared),
            key_squared
        ]
        value_components = [value, value_flat]
        if self.use_pair:
            pair_value = jnp.repeat(pair_b[:, None], self.heads, axis=1)
            query_components.append(bias_query)
            key_components.append(bias_key)
            value_components.append(pair_value)
        full_query = jnp.concatenate(query_components, axis=-1)
        full_key = jnp.concatenate(key_components, axis=-1)
        full_value = jnp.concatenate(value_components, axis=-1)
        # work around the jax implementation requiring equal key and value size
        truncate = False
        value_size = full_value.shape[-1]
        if full_value.shape[-1] != full_key.shape[-1]:
            truncate = True
            full_value = jnp.concatenate((
                full_value,
                jnp.zeros((local.shape[0], self.heads,
                           full_key.shape[-1] - value_size))), axis=-1)
        out_dtype = full_query.dtype
        # FIXME: cudnn backend does not work out of the box with pair bias
        if self.backend == "cudnn":
            full_query = full_query.astype(jnp.bfloat16)
            full_key = full_key.astype(jnp.bfloat16)
            full_value = full_value.astype(jnp.bfloat16)
            if pair_bias is not None:
                raise NotImplementedError("requires workaround for pair bias.")
        if pair_bias is not None:
            pair_bias = jnp.moveaxis(pair_bias, -1, 0)
        out_value = jax.nn.dot_product_attention(
            full_query,
            full_key,
            full_value,
            scale=1.0, mask=mask, bias=pair_bias,
            implementation=self.backend)
        out_value = out_value.astype(out_dtype)
        if truncate:
            out_value = out_value[..., :value_size]
        # process output
        if self.use_pair:
            out_scalar, out_points, out_pair = jnp.split(
                out_value, (value.shape[-1], value.shape[-1] + value_flat.shape[-1]), axis=-1)
            out_pair = jnp.einsum(
                "ikr,ihkr->ihk", pair_a, out_pair).reshape(query.shape[0], -1)
        else:
            out_scalar, out_points = jnp.split(
                out_value, (value.shape[-1],), axis=-1)
        out_scalar = out_scalar.reshape(local.shape[0], -1)
        out_points = Vec3Array.from_array(
            out_points.reshape(*out_points.shape[:-2], -1, 3))
        if self.equivariant:
            out_points = frames[:, None].apply_inverse_to_point(out_points)
        else:
            out_points += frames.translation[:, None]
        out_norm = out_points.norm().reshape(query.shape[0], -1)
        out_points = out_points.to_array().reshape(query.shape[0], -1)
        out_components = [out_scalar, out_points, out_norm]
        if self.use_pair:
            out_components.append(out_pair)
        result = Linear(local.shape[-1], initializer=self.final_init, bias=False)(
            jnp.concatenate(out_components, axis=-1))
        if self.return_attn:
            attn_components = dict(Q=full_query, K=full_key, V=full_value, bias=pair_bias)
            return result, attn_components
        return result

# FIXME: adjust usage in other modules
class HybridPointAttention(hk.Module):
    """Hybrid attention update using dense attention with light-weight
    features and sparse attention with heavy-weight features (distances, etc).
    """
    def __init__(self, name = "hybrid_attn", heads=8, **kwargs):
        super().__init__(name)
        self.kwargs = kwargs
        self.heads = heads

    def __call__(self, local, pos, pair_features, neighbours,
                 resi, chain, batch, mask,
                 pair_bias=None):
        neighbour_mask = mask[neighbours] * (neighbours != -1) > 0
        pair_mask = mask[:, None] * mask[None, :]
        pair_mask *= batch[:, None] == batch[None, :]
        pair_value = pair_features
        frames, _ = extract_aa_frames(Vec3Array.from_array(pos))
        update = FlashPointAttention(heads=self.heads, **self.kwargs)(
            local, frames.to_array(),
            pair_mask, pair_bias=pair_bias) # TODO: pair bias?
        attn = Linear(self.heads, bias=False)(pair_features)
        attn = masked_softmax(neighbour_mask[..., None], attn, axis=1)
        out = jnp.einsum("ijh,ijc->ihc", attn, pair_value)
        update += Linear(local.shape[-1], bias=False, initializer="zeros")(
                out.reshape(out.shape[0], -1))
        return update

def _pad_to_multiple(data, base=64):
    remainder = data.shape[-1] % base
    if remainder > 0:
        padding = base - remainder
        data = jnp.concatenate((
            data, jnp.zeros(list(data.shape[:-1]) + [padding], dtype=data.dtype)),
            axis=-1)
    return data
