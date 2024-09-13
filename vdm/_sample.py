from typing import Tuple, Sequence, Optional
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Key, Array

from ._sde import _alpha_sigma


def _decode(z_0_rescaled, gamma_0, key):
    n = jr.normal(key, z_0_rescaled.shape) * 1e-3
    return z_0_rescaled + n


def _generate_x(z_0, gamma_0, key):
    alpha_0, _ = _alpha_sigma(gamma_0)
    z_0_rescaled = z_0 / alpha_0
    sample = _decode(z_0_rescaled, gamma_0, key)
    return sample


@eqx.filter_jit
def sample_step(
    i: int, 
    vdm: eqx.Module, 
    T_sample: int, 
    z_t: Array, 
    key: Key, 
    sharding: Optional[jax.sharding.Sharding] = None
) -> Tuple[Array, Array]:

    key_eps, key_time = jr.split(jr.fold_in(key, i))
    keys_time = jnp.asarray(jr.split(key_time, len(z_t)))

    eps = jr.normal(key_eps, z_t.shape)

    t = (T_sample - i) / T_sample
    s = (T_sample - i - 1) / T_sample

    gamma_s = vdm.gamma(s)
    gamma_t = vdm.gamma(t)

    if sharding is not None:
        eps, z_t, keys_time = eqx.filter_shard(
            (eps, z_t, keys_time), sharding
        )

    score_fn = jax.vmap(vdm.score_network, in_axes=(0, None, 0))
    eps_hat = score_fn(z_t, gamma_t, keys_time)

    a = jax.nn.sigmoid(-gamma_s)
    b = jax.nn.sigmoid(-gamma_t)
    c = -jnp.expm1(gamma_s - gamma_t)
    sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_t))

    z_s = jnp.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) + jnp.sqrt((1. - a) * c) * eps

    alpha_t = jnp.sqrt(1. - b)
    x_pred = (z_t - sigma_t * eps_hat) / alpha_t

    return z_s, x_pred # Return both if not jax.lax.fori_loop


def sample_fn(
    key: Key, 
    vdm: eqx.Module, 
    N_sample: int, 
    T_sample: int, 
    data_shape: Sequence[int], 
    sharding: Optional[jax.sharding.Sharding] = None
) -> Tuple[Array, ...]:
    key_z, key_sample, key_loop = jr.split(key, 3)

    z = jr.normal(key_z, (N_sample,) + data_shape)
    if sharding is not None:
        z = jax.device_put(z, sharding)

    def _sample_step_i(i, z_t_and_x):
        z_t, _ = z_t_and_x
        key_i = jr.fold_in(key_loop, i)
        fn = lambda z_t, i: sample_step(
            i, 
            vdm, 
            T_sample, 
            z_t, 
            key_i,
            sharding
        )
        return fn(z_t, i) # z_t must be first argument

    z, x_pred = jax.lax.fori_loop(
        lower=0, 
        upper=T_sample, 
        body_fun=_sample_step_i, 
        init_val=(z, jnp.zeros_like(z))
    )

    key_samples = jr.split(key_sample, len(z))
    gamma_0 = vdm.gamma(0.)
    x_sample = jax.vmap(_generate_x, in_axes=(0, None, 0))(
        z, gamma_0, key_samples
    )
    return z, x_pred, x_sample