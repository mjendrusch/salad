
import jax
import jax.numpy as jnp

import haiku as hk

from salad.modules.utils.collections import dotdict
from salad.modules.basic import Linear, init_glorot

class DenseNEPointAttention(hk.Module):
    def __init__(self, config = None, key_size=32,
                 num_heads=8, num_points=0,
                 pair_value=False,
                 name = "ne_point_attention"):
        super().__init__(name)
        self.config = config or dotdict()
        self.config["num_heads"] = self.config.get("num_heads", None) or num_heads
        self.config["num_points"] = self.config.get("num_points", None) or num_points
        self.config["key_size"] = self.config.get("key_size", None) or key_size
        self.config["pair_value"] = self.config.get("pair_value", None) or pair_value

    def __call__(self, local, pair=None, pos=None, mask=None):
        c = self.config
        # query / key / value computation and reshaping
        def attention_component(x):
            result = Linear(self.key_size * self.heads, bias=True, initializer=init_glorot())(x)
            # sequence, heads, head_size
            return result.reshape(*result.shape[:-1], self.heads, self.key_size)
        def attention_points(x, pos=None):
            result = Linear(self.num_points * self.heads, bias=False, initializer="linear")(x).reshape(
                x.shape[0], self.heads, self.num_points, 3)
            if pos is not None:
                result += pos[:, None, None, :]
            return result
        query = hk.LayerNorm([-1], True, True)(attention_component(local))
        key = hk.LayerNorm([-1], True, True)(attention_component(local))
        value = attention_component(local)
        if c.num_points != 0:
            # flash point attention
            w_C = jnp.sqrt(2 / (9 * self.key_points))
            w_L = jnp.sqrt(1 / 3)
            query_points = attention_points(local, pos)
            key_points = attention_points(local, pos)
            value_points = attention_points(local, pos)
            # per-head scale factor for point-distance attention
            gamma = hk.get_parameter(
                "gamma", (self.heads,),
                init=hk.initializers.Constant(jnp.log(jnp.exp(1.) - 1.))
            )
            gamma = gamma[None, :, None]
            dfactor = jax.nn.softplus(gamma)
            # flatten query, key and value to put them in the correct shape for standard attention
            query_flat = query_points.to_array().reshape(*query.shape[:2], -1)
            query_squared = query_points.norm2()
            key_flat = key_points.to_array().reshape(*key.shape[:2], -1)
            key_squared = key_points.norm2()
            value_flat = value_points.to_array().reshape(*value.shape[:2], -1)

            # combine query and key features in such a way that taking the inner product
            # of query and key results in standard IPA (with factored pair bias)
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
            query = jnp.concatenate(query_components, axis=-1)
            key = jnp.concatenate(key_components, axis=-1)
            value = jnp.concatenate(value_components, axis=-1)
        bias = None
        if pair is not None:
            bias = Linear(c.num_heads, initializer="zeros")(pair)
        attn = jnp.einsum("iha,jhb->ijh", query, key) + bias - 1e9 * (1 - mask)
        attn = jax.nn.softmax(attn, axis=1)
        attn = jnp.where(mask[..., None], attn, 0)
        out = jnp.einsum("ijh,jhc->ihc", attn, value)
        if c.pair_value:
            pair_out = jnp.einsum("ijh,ijc->ihc", attn, pair).reshape(attn.shape[0], -1)
            out = jnp.concatenate((out, pair_out), axis=-1)
        out = out.reshape(out.shape[0], -1)
        out = Linear(local.shape[-1], initializer="zeros")(out)
        out *= jax.nn.sigmoid(Linear(local.shape[-1], bias=True)(local))
        return out

class SparseNEPointAttention(hk.Module):
    def __init__(self, config = None, key_size=32,
                 num_heads=8, num_points=0,
                 pair_value=False,
                 name = "ne_point_attention"):
        super().__init__(name)
        self.config = config or dotdict()
        self.config["num_heads"] = self.config.get("num_heads", None) or num_heads
        self.config["num_points"] = self.config.get("num_points", None) or num_points
        self.config["key_size"] = self.config.get("key_size", None) or key_size
        self.config["pair_value"] = self.config.get("pair_value", None) or pair_value

    def __call__(self, local, pair=None, pos=None, neighbours=None):
        c = self.config
        # query / key / value computation and reshaping
        def attention_component(x):
            result = Linear(c.key_size * c.num_heads, bias=True, initializer=init_glorot())(x)
            # sequence, heads, head_size
            return result.reshape(*result.shape[:-1], c.num_heads, c.key_size)
        def attention_points(x, pos=None):
            result = Linear(c.num_points * c.num_heads, bias=False, initializer="linear")(x).reshape(
                x.shape[0], c.num_heads, c.num_points, 3)
            if pos is not None:
                result += pos[:, None, None, :]
            return result
        query = hk.LayerNorm([-1], True, True)(attention_component(local))
        key = hk.LayerNorm([-1], True, True)(attention_component(local))
        value = attention_component(local)
        if c.num_points != 0:
            # flash point attention
            w_C = jnp.sqrt(2 / (9 * self.key_points))
            w_L = jnp.sqrt(1 / 3)
            query_points = attention_points(local, pos)
            key_points = attention_points(local, pos)
            value_points = attention_points(local, pos)
            # per-head scale factor for point-distance attention
            gamma = hk.get_parameter(
                "gamma", (self.heads,),
                init=hk.initializers.Constant(jnp.log(jnp.exp(1.) - 1.))
            )
            gamma = gamma[None, :, None]
            dfactor = jax.nn.softplus(gamma)
            # flatten query, key and value to put them in the correct shape for standard attention
            query_flat = query_points.to_array().reshape(*query.shape[:2], -1)
            query_squared = query_points.norm2()
            key_flat = key_points.to_array().reshape(*key.shape[:2], -1)
            key_squared = key_points.norm2()
            value_flat = value_points.to_array().reshape(*value.shape[:2], -1)

            # combine query and key features in such a way that taking the inner product
            # of query and key results in standard IPA (with factored pair bias)
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
            query = jnp.concatenate(query_components, axis=-1)
            key = jnp.concatenate(key_components, axis=-1)
            value = jnp.concatenate(value_components, axis=-1)
        mask = (neighbours != -1)
        bias = None
        if pair is not None:
            bias = Linear(c.num_heads, initializer="zeros")(pair)
        attn = jnp.einsum("iha,ijhb->ijh", query, key[neighbours]) + bias - 1e9 * (1 - mask)
        attn = jax.nn.softmax(attn, axis=1)
        attn = jnp.where(mask[..., None], attn, 0)
        out = jnp.einsum("ijh,ijhc->ihc", attn, value[neighbours])
        if c.pair_value:
            pair_out = jnp.einsum("ijh,ijc->ihc", attn, pair).reshape(attn.shape[0], -1)
            out = jnp.concatenate((out, pair_out), axis=-1)
        out = out.reshape(out.shape[0], -1)
        out = Linear(local.shape[-1], initializer="zeros")(out)
        out *= jax.nn.sigmoid(Linear(local.shape[-1], bias=True)(local))
        return out

