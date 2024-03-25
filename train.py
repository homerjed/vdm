from typing import Tuple
import jax 
import jax.random as jr 
import jax.numpy as jnp
import equinox as eqx
import optax

from vdm import vlb


def loss_fn(
    vdm: eqx.Module, 
    key: jr.PRNGKey, 
    x: jax.Array, 
    t: float | jax.Array, 
    shard: jax.sharding.Sharding | None = None
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]:
    loss, metrics = vlb(vdm, x, key, t, shard)
    return loss, metrics


def sample_times(key, n):
    t = jr.uniform(key, (n,), minval=0., maxval=1. / n)
    t = t + (1. / n) * jnp.arange(n)
    return t


@eqx.filter_jit
def batch_loss_fn(
    vdm: eqx.Module, 
    key: jr.PRNGKey, 
    x: jax.Array, 
    shard: jax.sharding.Sharding | None = None
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]: 
    key, key_t = jr.split(key)
    n = len(x)
    keys = jr.split(key, n)
    t = sample_times(key_t, n)
    # Antithetic time sampling for lower variance VLB(x)
    _fn = eqx.filter_vmap(loss_fn, in_axes=(None, 0, 0, 0, None))
    loss, metrics = _fn(vdm, keys, x, t, shard)
    return loss.mean(), [m.mean() for m in metrics]


@eqx.filter_jit
def make_step(
    vdm: eqx.Module, 
    x: jax.Array, 
    key: jr.PRNGKey, 
    opt_state: optax.OptState, 
    opt_update: optax.GradientTransformation,
    shard: jax.sharding.Sharding | None = None
) -> Tuple[
        eqx.Module, 
        jax.Array, 
        Tuple[jax.Array, jax.Array, jax.Array], 
        optax.OptState
    ]:
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn, has_aux=True)
    (loss, metrics), grads = loss_fn(vdm, key, x, shard)
    updates, opt_state = opt_update(grads, opt_state, vdm)
    vdm = eqx.apply_updates(vdm, updates)
    return vdm, loss, metrics, opt_state