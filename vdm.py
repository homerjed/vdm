from typing import Tuple
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.scipy.stats as jss
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd


def _var(gamma):
    return jax.nn.sigmoid(gamma)


def _alpha_sigma(gamma):
    var = _var(gamma)
    return jnp.sqrt(1. - var), jnp.sqrt(var)


def _encode(x):
    return x #(x - x.mean()) / x.std()


def _decode(z_0_rescaled, gamma_0, key):
    # dist = tfd.Independent(
    #     tfd.Normal(
    #         z_0_rescaled, jnp.array([1e-3])
    #     ),  #jnp.exp(0.5 * gamma_0)),
    #     reinterpreted_batch_ndims=3
    # )
    # return dist.sample(seed=key)
    return jr.normal(key, z_0_rescaled.shape) * 1e-3


def _logprob(x, z_0_rescaled, gamma_0):
    # dist = tfd.Independent(
    #     tfd.Normal(
    #         z_0_rescaled, jnp.array([1e-3])
    #     ),  #jnp.exp(0.5 * gamma_0)),
    #     reinterpreted_batch_ndims=3
    # )
    # return dist.log_prob(x)
    return jss.norm.pdf(x, loc=z_0_rescaled, scale=1e-3)


def _generate_x(z_0, gamma_0, key):
    alpha_0, _ = _alpha_sigma(gamma_0)
    z_0_rescaled = z_0 / alpha_0
    sample = _decode(z_0_rescaled, gamma_0, key)
    return sample


def get_gamma_alpha_sigma(vdm, t):
    gamma_t = vdm.gamma(t)
    return gamma_t, *_alpha_sigma(gamma_t)


def vlb(
    vdm: eqx.Module, 
    x: jax.Array, 
    key: jr.PRNGKey, 
    t: float | jax.Array = None, 
    # T_train: float = 0.,
    shard=jax.sharding.Sharding | None
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
    key, key_0, key_d = jr.split(key, 3)

    gamma_0, alpha_0, sigma_0 = get_gamma_alpha_sigma(vdm, 0.)
    gamma_t, alpha_t, sigma_t = get_gamma_alpha_sigma(vdm, t)
    gamma_1, alpha_1, sigma_1 = get_gamma_alpha_sigma(vdm, 1.)

    # encode
    f = _encode(x)

    # 1. RECONSTRUCTION LOSS
    # add noise and reconstruct
    eps_0 = jr.normal(key_0, shape=f.shape)
    eps_0 = jax.device_put(eps_0, shard)
    z_0 = alpha_0 * f + sigma_0 * eps_0
    z_0_rescaled = f + jnp.exp(0.5 * gamma_0) * eps_0  # = z_0/sqrt(1-var)
    loss_recon = -_logprob(f, z_0_rescaled, gamma_0)

    # 2. LATENT LOSS
    # KL z1 with N(0,1) prior
    mean1_sqr = jnp.square(alpha_1 * f)
    loss_klz = 0.5 * jnp.sum(mean1_sqr + sigma_1 ** 2. - jnp.log(sigma_1 ** 2.) - 1.)

    # 3. DIFFUSION LOSS
    # discretize time steps if we're working with discrete time
    # if T_train > 0:
    #     t = jnp.ceil(t * T_train) / T_train

    # sample z_t 
    eps = jr.normal(key_d, shape=f.shape)
    eps = jax.device_put(eps, shard)
    z_t = alpha_t * f + sigma_t * eps
    # compute predicted noise
    key, _ = jr.split(key)
    eps_ = vdm.score(z_t, gamma_t, key=key)
    # compute MSE of predicted noise
    loss_diff_mse = jnp.square(eps - eps_).sum()

    # if T_train == 0:
    # loss for infinite depth T, i.e. continuous time
    _, g_t_grad = jax.jvp(vdm.gamma, (t,), (jnp.ones_like(t),))
    loss_diff = 0.5 * g_t_grad * loss_diff_mse
    # else:
    #     # loss for finite depth T, i.e. discrete time
    #     s = t - (1. / T_train)
    #     gamma_s = vdm.gamma(s)
    #     loss_diff = 0.5 * T_train * jnp.expm1(gamma_t - gamma_s) * loss_diff_mse

    loss = loss_klz + loss_recon + loss_diff
    metrics = [loss_klz, loss_recon, loss_diff]
    return loss, metrics