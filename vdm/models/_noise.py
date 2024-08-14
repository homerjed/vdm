from typing import Optional, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array


class NoiseSchedule(eqx.Module):
    w: Array
    b: Array

    def __init__(self, init_gamma_0, init_gamma_1):
        init_bias = init_gamma_0
        init_scale = init_gamma_1 - init_gamma_0
        self.w = init_scale
        self.b = init_bias

    def __call__(self, t: Array) -> Array:
        return abs(self.w) * t + self.b


class LinearMonotone(eqx.Module):
    weight: Optional[Array] = None
    bias: Optional[Array] = None
    use_bias: bool = True

    def __init__(self, in_size=None, out_size=None, use_bias=True, *, key):
        self.use_bias = use_bias
        key_w, key_b = jr.split(key)
        self.weight = jr.normal(key_w, (out_size, in_size))
        if self.use_bias:
            self.bias = jnp.zeros((out_size,))

    def __call__(self, x: Array) -> Array:
        x = jnp.abs(self.weight) @ x 
        if self.use_bias:
            x = x + self.bias
        return x


class NoiseScheduleNN(eqx.Module):
    n_features: int = 128 
    nonlinear: bool = True
    weight: Array
    bias: Array
    l2: LinearMonotone
    l3: LinearMonotone

    def __init__(self, gamma_0: Array, gamma_1: Array, *, key: Key):
        """ Montonically increasing linear layer, sigmoid in call => bias made positive """
        init_bias = gamma_0
        init_scale = gamma_1 - init_bias

        # Baseline linear schedule parameters
        self.weight = jnp.array([init_scale])
        self.bias = jnp.array([init_bias])

        # Non-linear schedule parameters
        key_2, key_3 = jr.split(key)
        if self.nonlinear:
            self.l2 = LinearMonotone(
                in_size=1, out_size=self.n_features, key=key_2
            )
            self.l3 = LinearMonotone(
                in_size=self.n_features, out_size=1, use_bias=False, key=key_3
            )

    def __call__(self, t: Union[float, Array]) -> Array:
        t = jnp.atleast_1d(t)
        h = jnp.abs(self.weight) @ t + self.bias
        if self.nonlinear:
            _h = 2. * (t - .5)  # scale input to [-1, +1]
            _h = self.l2(_h)
            _h = 2 * (jax.nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
            _h = self.l3(_h) / self.n_features
            h += _h
        return h