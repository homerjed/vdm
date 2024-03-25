import jax
import jax.numpy as jnp


def _var(gamma):
    return jax.nn.sigmoid(gamma)


def _alpha_sigma(gamma):
    var = _var(gamma)
    return jnp.sqrt(1. - var), jnp.sqrt(var)