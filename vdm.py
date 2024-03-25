from typing import Tuple
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.scipy.stats as jss
import equinox as eqx

from sde import _alpha_sigma


def _encode(x):
    return x 


def _logprob(x, z_0_rescaled, gamma_0):
    return jss.norm.pdf(x, loc=z_0_rescaled, scale=1e-3)


def _get_gamma_alpha_sigma(vdm, t):
    gamma_t = vdm.gamma(t)
    return gamma_t, *_alpha_sigma(gamma_t)


def vlb(
    vdm: eqx.Module, 
    x: jax.Array, 
    key: jr.PRNGKey, 
    t: float | jax.Array = None, 
    shard: jax.sharding.Sharding | None = None
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:

    key, key_0, key_d, key_e = jr.split(key, 4)

    # Diffusion parameters
    gamma_0, alpha_0, sigma_0 = _get_gamma_alpha_sigma(vdm, 0.)
    gamma_t, alpha_t, sigma_t = _get_gamma_alpha_sigma(vdm, t)
    gamma_1, alpha_1, sigma_1 = _get_gamma_alpha_sigma(vdm, 1.)

    # Encode
    f = _encode(x)

    # 1. RECONSTRUCTION LOSS
    # Add noise and reconstruct
    eps_0 = jr.normal(key_0, shape=f.shape)
    if shard is not None:
        eps_0 = jax.device_put(eps_0, shard)

    z_0 = alpha_0 * f + sigma_0 * eps_0
    z_0_rescaled = f + jnp.exp(0.5 * gamma_0) * eps_0  # = z_0/sqrt(1-var)
    loss_recon = -_logprob(f, z_0_rescaled, gamma_0)

    # 2. LATENT LOSS
    # KL z1 with N(0,1) prior
    mean1_sqr = jnp.square(alpha_1 * f)
    loss_klz = 0.5 * jnp.sum(mean1_sqr + sigma_1 ** 2. - jnp.log(sigma_1 ** 2.) - 1.)

    # 3. DIFFUSION LOSS
    # Sample z_t 
    eps = jr.normal(key_d, shape=f.shape)
    if shard is not None:
        eps = jax.device_put(eps, shard)

    # Marginal sample
    z_t = alpha_t * f + sigma_t * eps

    # Compute predicted noise
    eps_ = vdm.score(z_t, gamma_t, key=key_e)
    # Compute MSE of predicted noise
    loss_diff_mse = jnp.square(eps - eps_).sum()

    # Loss for infinite depth T, i.e. continuous time
    _, g_t_grad = jax.jvp(vdm.gamma, (t,), (jnp.ones_like(t),))
    loss_diff = 0.5 * g_t_grad * loss_diff_mse

    loss = loss_klz + loss_recon + loss_diff
    metrics = [loss_klz, loss_recon, loss_diff]
    return loss, metrics