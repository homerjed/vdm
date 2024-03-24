from typing import Tuple, Sequence
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import tensorflow_probability.substrates.jax.distributions as tfd


def _var(gamma):
    return jax.nn.sigmoid(gamma)


def _alpha_sigma(gamma):
    var = _var(gamma)
    return jnp.sqrt(1. - var), jnp.sqrt(var)


def _decode(z_0_rescaled, gamma_0, key):
    dist = tfd.Independent(
        tfd.Normal(
            z_0_rescaled, jnp.array([1e-3])
        ),  #jnp.exp(0.5 * gamma_0)),
        reinterpreted_batch_ndims=3
    )
    return dist.sample(seed=key)


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
    z_t: jax.Array, 
    key: jr.PRNGKey, 
    sharding: jax.sharding.Sharding
) -> Tuple[jax.Array, jax.Array]:
    key = jr.fold_in(key, i)
    eps = jr.normal(key, z_t.shape)
    if sharding is not None:
        eps = jax.device_put(eps, sharding)

    t = (T_sample - i) / T_sample
    s = (T_sample - i - 1) / T_sample

    gamma_s = vdm.gamma(s)
    gamma_t = vdm.gamma(t)

    keys = jr.split(key, len(z_t))
    if sharding is not None:
        z_t, gamma_t, keys = jax.device_put(
            (z_t, gamma_t, keys), sharding
        )
    # Was using vdm.score_network, be consistent with methods
    _fn = jax.vmap(vdm.score_network, in_axes=(0, None, 0))
    eps_hat = _fn(z_t, gamma_t, keys)

    a = jax.nn.sigmoid(-gamma_s)
    b = jax.nn.sigmoid(-gamma_t)
    c = -jnp.expm1(gamma_s - gamma_t)
    sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_t))

    z_s = jnp.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) + jnp.sqrt((1. - a) * c) * eps

    alpha_t = jnp.sqrt(1. - b)
    x_pred = (z_t - sigma_t * eps_hat) / alpha_t

    return z_s, x_pred # return both if not jax.lax.fori_loop


def sample_fn(
    key: jr.PRNGKey, 
    vdm: eqx.Module, 
    N_sample: int, 
    T_sample: int, 
    data_shape: Sequence[int], 
    sharding: jax.sharding.Sharding
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    print("Sampling...")

    z = jr.normal(key, (N_sample,) + data_shape)
    if sharding is not None:
        z = jax.device_put(z, sharding)

    def body_fn(i, z_t_and_x):
        z_t, _ = z_t_and_x
        fn = lambda z_t, i: sample_step(
            i, vdm, T_sample, z_t, key, sharding
        )
        return fn(z_t, i) # z_t must be first argument

    z, x_pred = jax.lax.fori_loop(
        lower=0, upper=T_sample, body_fun=body_fn, init_val=(z, z)
    )
    key, _ = jr.split(key)
    x_sample = jax.vmap(_generate_x, in_axes=(0, None, None))(
        z, vdm.gamma(0.), key
    )
    print("...sampled.")
    print("z, x_pred, x_sample", z.shape, x_pred.shape, x_sample.shape)
    return z, x_pred, x_sample


