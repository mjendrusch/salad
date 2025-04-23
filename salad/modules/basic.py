from typing import Union, Optional

import jax
import jax.numpy as jnp
import haiku as hk

from salad.aflib.model.layer_stack import layer_stack

class Linear(hk.Module):
    def __init__(self,
                 size: int=128,
                 bias: bool=True,
                 bias_init: float=0.0,
                 initializer: Union[str, hk.initializers.Initializer]="linear",
                 name: Optional[str]="linear"):
        r"""Linear layer compatible with initialisation choices made in AlphaFold.
        
        Args:
            size (int): size of the output representation.
            bias (bool): add bias to output? Default: True.
            bias_init (float): value to initialise bias to. Default: 0.0.
            initializer (str | Initializer): Initialisation of module weights.
                One of "relu" (for relu activation), "linear" (for linear activation)
                and "zeros" (for zero initialisation). Alternatively, an Initializer
                may be specified.
            name (str): module name.
        """
        super().__init__(name=name)
        self.size = size
        self.bias = bias
        self.initializer = initializer
        self.bias_init = bias_init

    @property
    def weight_init(self):
        if isinstance(self.initializer, str):
            if self.initializer == "relu":
                return init_relu()
            if self.initializer == "glorot":
                return init_glorot()
            if self.initializer == "linear":
                return init_linear()
            if self.initializer == "zeros":
                return init_zeros()
            return init_linear()
        else:
          return self.initializer

    def __call__(self, data):
        w = hk.get_parameter("w", [data.shape[-1], self.size], init=self.weight_init)
        b = hk.get_parameter("b", [self.size], init=hk.initializers.Constant(self.bias_init))

        out = jnp.einsum("...k,ki->...i", data, w)
        if self.bias:
            out = out + b
        return out

class MLP(hk.Module):
    def __init__(self, size=64, out_size=None, depth=2,
                 activation=jax.nn.relu, bias=True,
                 init="relu", final_init="linear",
                 name: Optional[str] = "mlp"):
        super().__init__(name)
        self.size = size
        self.out_size = out_size or self.size
        self.depth = depth
        self.activation = activation
        self.bias = bias
        self.init = init
        self.final_init = final_init

    def __call__(self, inputs):
        out = inputs
        for idx in range(self.depth):
            if idx < self.depth - 1:
                out = Linear(
                    self.size, initializer=self.init, bias=self.bias)(out)
                out = self.activation(out)
            else:
                out = Linear(
                    self.out_size, initializer=self.final_init, bias=self.bias)(out)
        return out

class GatedMLP(hk.Module):
    def __init__(self, size=64, out_size=None,
                 activation=jax.nn.gelu,
                 init="relu",
                 final_init="zeros",
                 name: Optional[str] = "gated_mlp"):
        super().__init__(name)
        self.size = size
        self.out_size = out_size
        self.activation = activation
        self.init = init
        self.final_init = final_init

    def __call__(self, inputs):
        out_size = self.out_size or inputs.shape[-1]
        gate = self.activation(Linear(
            self.size, bias=False, initializer=self.init)(inputs))
        hidden = gate * Linear(
            self.size, bias=False, initializer=self.init)(inputs)
        return Linear(
            out_size, bias=False, initializer=self.final_init)(hidden)

def init_relu():
    return hk.initializers.VarianceScaling(mode="fan_in", scale=2.0)

def init_linear():
    return hk.initializers.VarianceScaling(mode="fan_in", scale=1.0)

def init_zeros():
    return hk.initializers.Constant(0.0)

def init_glorot():
  return hk.initializers.VarianceScaling(
    mode="fan_avg", scale=1.0, distribution="uniform")

def init_small(scale=1e-2):
  return hk.initializers.RandomUniform(minval=-scale, maxval=scale)

def small_linear(*args, scale=1e-4, **kwargs):
  def inner(x):
    return hk.LayerNorm([-1], False, False)(
      Linear(*args, **kwargs, initializer=init_small(scale=scale))(x))
  return inner

def block_stack(depth, block_size=1, with_state=False):
    count = depth // block_size
    def inner(function):
        if block_size > 1:
            block = hk.remat(layer_stack(block_size, with_state=with_state)(function))
            return layer_stack(count, with_state=with_state)(block)
        return layer_stack(count, with_state=with_state)(function)
    return inner
