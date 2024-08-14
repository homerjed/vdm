from typing import Optional, Union
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Key, Array

from ._unet import UNet
from ._mixer import Mixer2d
from ._noise import NoiseScheduleNN


class VDM(eqx.Module):
    score_network: eqx.Module
    noise_network: eqx.Module

    def __init__(self, score_network, noise_network):
        self.score_network = score_network
        self.noise_network = noise_network
    
    def score(self, x, gamma_t, *, key):
        gamma_t = jnp.atleast_1d(gamma_t)
        return self.score_network(x, gamma_t, key=key)

    def gamma(self, t):
        return self.noise_network(t)


class cVDM(VDM):

    def __call__(self, x, y, t):
        gamma_t = self.noise_network(t)
        return self.score_network(x, y, gamma_t)
    
    def score(self, x, y, gamma_t):
        return self.score_network(x, y, jnp.atleast_1d(gamma_t))


class FourierFeatures(eqx.Module):
    def __call__(self, inputs: Array) -> Array:
        freqs = jnp.asarray(range(2, 8), dtype=inputs.dtype) #[0, 1, ..., 7]
        w = 2. ** freqs * 2. * jnp.pi
        w = jnp.tile(w, (inputs.size,))
        h = jnp.repeat(inputs, len(freqs), axis=-1)
        h *= w
        h = jnp.concatenate([jnp.sin(h), jnp.cos(h)])
        return h


class ScoreNetwork(eqx.Module):
    gamma_0: float
    gamma_1: float
    net: eqx.Module

    def __init__(self, data_shape, context_dim, gamma_0, gamma_1, *, key):
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        # self.net = Mixer2d(
        #     data_shape,
        #     patch_size=4,
        #     hidden_size=64,
        #     mix_patch_size=64,
        #     mix_hidden_size=64,
        #     num_blocks=3,
        #     context_dim=context_dim,
        #     key=key
        # )
        self.net = UNet(
            data_shape=data_shape,
            is_biggan=False,
            dim_mults=[1, 2, 4, 8],
            hidden_size=256,
            heads=4,
            dim_head=32,
            dropout_rate=0.,
            num_res_blocks=2,
            attn_resolutions=[16],
            key=key
        )

    def __call__(self, z, gamma_t, key):
        _gamma_t = 2. * (gamma_t - self.gamma_0) / (self.gamma_1 - self.gamma_0) - 1.
        h = self.net(_gamma_t, z, key=key)
        return h