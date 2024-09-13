from typing import Tuple, Union, Optional
import os
import jax 
import jax.random as jr 
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Key, Array
import optax
from tqdm import trange

from ._vlb import vlb
from ._utils import unbatch, save_opt_state_and_model
from ._sample import sample_fn
from ._plots import plot_metrics, plot_samples


def loss_fn(
    vdm: eqx.Module, 
    key: Key, 
    x: Array, 
    t: Union[float, Array], 
    shard: Optional[jax.sharding.Sharding] = None
) -> Tuple[Array, Tuple[Array, ...]]:
    loss, metrics = vlb(vdm, key, x, t, shard)
    return loss, metrics


def sample_times(key: Key, n: int) -> Array:
    t = jr.uniform(key, (n,), minval=0., maxval=1. / n)
    t = t + (1. / n) * jnp.arange(n)
    return t


@eqx.filter_jit
def batch_loss_fn(
    vdm: eqx.Module, 
    key: Key, 
    x: Array, 
    shard: Optional[jax.sharding.Sharding] = None
) -> Tuple[Array, Tuple[Array, ...]]: 
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
    key: Key, 
    x: Array, 
    opt_state: optax.OptState, 
    opt_update: optax.GradientTransformation,
    shard: Optional[jax.sharding.Sharding] = None
) -> Tuple[eqx.Module, Array, Tuple[Array, ...], optax.OptState]:
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn, has_aux=True)
    (loss, metrics), grads = loss_fn(vdm, key, x, shard)
    updates, opt_state = opt_update(grads, opt_state, vdm)
    vdm = eqx.apply_updates(vdm, updates)
    return vdm, loss, metrics, opt_state


def train(
    key, 
    vdm, 
    dataset, 
    opt, 
    opt_state, 
    n_steps, 
    n_batch, 
    shard,
    n_sample,
    T_sample,
    model_name="vdm",
    imgs_dir=None
):
    key, sample_key = jr.split(key)

    data_shape = dataset.data_shape

    losses = []
    metrics = []
    with trange(n_steps, colour="blue") as steps:
        for s, train_batch, valid_batch in zip(
            steps,
            dataset.train_dataloader.loop(n_batch),
            dataset.valid_dataloader.loop(n_batch)
        ):
            key, train_key, valid_key = jr.split(key, 3)

            # Train
            x, y = unbatch(train_batch, shard)
            vdm, loss_train, train_metrics, opt_state = make_step(
                vdm, train_key, x, opt_state, opt.update, shard
            )

            # Validate
            x, y = unbatch(valid_batch, shard)
            loss_valid, valid_metrics = batch_loss_fn(vdm, valid_key, x, shard)

            # Record
            losses += [(loss_train, loss_valid)]
            metrics += [(train_metrics, valid_metrics)]
            steps.set_postfix(
                Lt=f"{losses[-1][0]:.3E}", Lv=f"{losses[-1][1]:.3E}"
            )

            # Sample
            if s % 5_000 == 0:
                _, _, samples = sample_fn(
                    sample_key, vdm, n_sample, T_sample, data_shape, shard
                )
                samples = dataset.scaler.reverse(samples) # [-1, 1] -> [0, 1]

                if imgs_dir is not None:
                    plot_samples(
                        samples, filename=os.path.join(imgs_dir, f"samples_{s:06d}.png")
                    )

                    plot_metrics(
                        vdm, losses, metrics, filename=os.path.join(imgs_dir, "loss.png")
                    )

                save_opt_state_and_model(
                    opt_state=opt_state,
                    model=vdm,
                    filename_opt=model_name + "_opt.eqx",
                    filename_model=model_name + ".eqx"
                )
    return vdm 