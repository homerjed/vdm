from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .unet import UNet

Array = jnp.ndarray


class VDM(eqx.Module):
    score_network: eqx.Module
    noise_network: eqx.Module

    def __init__(self, score_network, noise_network):
        self.score_network = score_network
        self.noise_network = noise_network

    def __call__(self, x, t, *, key):
        gamma_t = self.noise_network(t)
        return self.score_network(x, t, key=key)
    
    def score(self, x, gamma_t, *, key):
        return self.score_network(x, jnp.atleast_1d(gamma_t), key=key)

    def gamma(self, t):
        return self.noise_network(t)


class cVDM(VDM):

    def __call__(self, x, y, t):
        gamma_t = self.noise_network(t)
        return self.score_network(x, y, gamma_t)
    
    def score(self, x, y, gamma_t):
        return self.score_network(x, y, jnp.atleast_1d(gamma_t))


class FourierFeatures(eqx.Module):
    def __call__(self, inputs):
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

# class ScoreNetwork(eqx.Module):
#     gamma_0: float
#     gamma_1: float
#     fourier_features: Optional[eqx.Module] = None
#     net: eqx.nn.MLP

#     def __init__(self, in_size, width_size, depth, activation, gamma_0, gamma_1, fourier_features, *, key):
#         key, _key = jr.split(key)
#         self.gamma_0 = gamma_0
#         self.gamma_1 = gamma_1
#         self.fourier_features = fourier_features
#         self.net = eqx.nn.MLP(
#             39, #in_size + 1 if fourier_features is not None else 39 #in_size + 1 + (in_size + 1) * 6 * 2, 
#             in_size, 
#             width_size, 
#             depth, 
#             activation=activation, 
#             key=_key)

#     def __call__(self, z, gamma_t):
#         _gamma_t = 2. * (gamma_t - self.gamma_0) / (self.gamma_1 - self.gamma_0) - 1.
#         h = jnp.concatenate([z, _gamma_t]) 
#         if self.fourier_features is not None:
#             h_ff = self.fourier_features(h)
#             h = jnp.concatenate([h, h_ff])
#         h = self.net(h)
#         return h


class NoiseSchedule(eqx.Module):
    w: Array
    b: Array

    def __init__(self, init_gamma_0, init_gamma_1):
        init_bias = init_gamma_0
        init_scale = init_gamma_1 - init_gamma_0
        self.w = init_scale
        self.b = init_bias

    def __call__(self, t):
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

    def __call__(self, x):
        x = jnp.abs(self.weight) @ x 
        if self.use_bias:
            x = x + self.bias
        return x


class NoiseScheduleNN(eqx.Module):
    n_features: int = 128 
    nonlinear: bool = True
    weight: Array
    bias: Array
    l2: eqx.Module
    l3: eqx.Module

    def __init__(self, gamma_0, gamma_1, *, key):
        """ Montonically increasing linear layer, sigmoid in call => bias made positive """
        init_bias = gamma_0
        init_scale = gamma_1 - init_bias

        # Baseline linear schedule parameters
        key, key_w, key_b = jr.split(key, 3)
        self.weight = jnp.array([init_scale])
        self.bias = jnp.array([init_bias])

        # Non-linear schedule parameters
        key, key_2, key_3 = jr.split(key, 3)
        if self.nonlinear:
            self.l2 = LinearMonotone(
                in_size=1, out_size=self.n_features, key=key_2)
            self.l3 = LinearMonotone(
                in_size=self.n_features, out_size=1, use_bias=False, key=key_3)

    def __call__(self, t):
        t = jnp.atleast_1d(t)
        h = jnp.abs(self.weight) @ t + self.bias
        if self.nonlinear:
            _h = 2. * (t - .5)  # scale input to [-1, +1]
            _h = self.l2(_h)
            _h = 2 * (jax.nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
            _h = self.l3(_h) / self.n_features
            h += _h
        return h