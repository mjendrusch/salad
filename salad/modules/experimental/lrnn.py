from typing import Optional
import jax
import jax.numpy as jnp
import haiku as hk
from salad.modules.basic import Linear, init_linear, init_zeros

def lrnn_apply(gate, update):
    def body(x, y):
        gate = x["gate"] * y["gate"]
        state = y["gate"] * x["state"] + y["state"]
        return dict(gate=gate, state=state)
    return jax.lax.associative_scan(body, dict(gate=gate, state=update))

def lrnn_apply_index(gate, update, index):
    # assume sorted index
    reset_index = jnp.roll(index, 1) != index
    # zero out sum of previous states at each reset index
    gate = gate.at[reset_index].set(0)
    return lrnn_apply(gate, update)

def bilrnn_apply(l_gate, l_update, r_gate, r_update, index):
    # assume sorted index
    l_reset = jnp.roll(index, 1) != index
    r_reset = jnp.roll(l_reset[::-1], 1)
    # zero out sum of previous states at each reset index
    gate = jnp.concatenate((l_gate.at[l_reset].set(0),
                            r_gate[::-1].at[r_reset]), axis=-1)
    update = jnp.concatenate((l_update, r_update[::-1]), axis=-1)
    return lrnn_apply(gate, update)

class BiLRNN(hk.Module):
    def __init__(self, size=2, state_size=16,
                 name: Optional[str] = "bilrnn"):
        super().__init__(name)
        self.size = size
        self.state_size = state_size

    def __call__(self, data, index):
        decay_bias = jnp.exp(hk.get_parameter("decay_bias",
                                              initializer=hk.initializers.RandomUniform(
                                                  minval=jnp.log(0.001),
                                                  maxval=jnp.log(1.0))))
        gate = jnp.exp(-jnp.exp(Linear(self.state_size, bias=False,
                                       initializer=init_linear())(data)) - decay_bias)
        gate_project = Linear(self.state_size, bias=False,
                              initializer=init_linear())(data)
        gate_channel = Linear(self.size, bias=False,
                              initializer=init_linear())(data)
        gate = gate[..., None, :] * gate_channel[..., :, None]
        value = Linear(self.size, bias=False,
                       initializer=init_linear())(data)[..., None]
        value *= Linear(self.state_size, bias=False, initializer=init_linear())(data)[..., None, :]
        l_gate, r_gate = jnp.split(gate, 2, axis=-1)
        l_value, r_value = jnp.split(value, 2, axis=-1)
        out = bilrnn_apply(l_gate, l_value, r_gate, r_value, index)
        out = jnp.einsum("...ck,...k->...c", out, gate_project)
        return out

class BiLRNNMLP(hk.Module):
    def __init__(self, factor=2, state_size=8, name: Optional[str] = "bilrnn_mlp"):
        super().__init__(name)
        self.factor = factor
        self.state_size = state_size

    def __call__(self, data, index):
        gate = jax.nn.gelu(Linear(data.shape[-1] * self.factor,
                                  bias=False, initializer=init_linear())(data))
        out = gate * BiLRNN(data, index)
        out = Linear(data.shape[-1], bias=False, initializer=init_zeros())(out)
        return out
