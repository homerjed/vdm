import time
import math
import os
import jax
import jax.numpy as jnp
import jax.random as jr 
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import equinox as eqx
import einops
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tensorflow_probability.substrates.jax import distributions as tfd

from models import VDM, NoiseScheduleNN, Mixer2d, UNet


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
            hidden_size=128,
            heads=4,
            dim_head=32,
            dropout_rate=0.,
            num_res_blocks=2,
            attn_resolutions=[16],
            key=key,
        )

    def __call__(self, z, gamma_t, key):
        _gamma_t = 2. * (gamma_t - self.gamma_0) / (self.gamma_1 - self.gamma_0) - 1.
        h = self.net(_gamma_t, z, key=key)
        return h

def _var(gamma):
    return jax.nn.sigmoid(gamma)

def _alpha_sigma(gamma):
    var = _var(gamma)
    return jnp.sqrt(1. - var), jnp.sqrt(var)

def data_decode(z_0_rescaled, gamma_0, key):
    dist = tfd.Independent(
        tfd.Normal(
            z_0_rescaled, 
            jnp.array([1e-3])
        ),  #jnp.exp(0.5 * gamma_0)),
        reinterpreted_batch_ndims=3
    )
    return dist.sample(seed=key)

def data_logprob(x, z_0_rescaled, gamma_0):
    dist = tfd.Independent(
        tfd.Normal(
            z_0_rescaled, 
            jnp.array([1e-3])
        ),  #jnp.exp(0.5 * gamma_0)),
        reinterpreted_batch_ndims=3
    )
    return dist.log_prob(x)

def data_generate_x(z_0, gamma_0, key):
    alpha_0, _ = _alpha_sigma(gamma_0)
    z_0_rescaled = z_0 / alpha_0
    sample = data_decode(z_0_rescaled, gamma_0, key)
    return sample

@eqx.filter_jit
def sample_step(i, vdm, T_sample, z_t, key, sharding):
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
        z_t, gamma_t, keys = jax.device_put((z_t, gamma_t, keys), sharding)
    # Was using vdm.score_network, be consistent with methods
    eps_hat = jax.vmap(vdm.score_network, in_axes=(0, None, 0))(z_t, gamma_t, keys)

    a = jax.nn.sigmoid(-gamma_s)
    b = jax.nn.sigmoid(-gamma_t)
    c = -jnp.expm1(gamma_s - gamma_t)
    sigma_t = jnp.sqrt(jax.nn.sigmoid(gamma_t))

    z_s = jnp.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) + jnp.sqrt((1. - a) * c) * eps

    alpha_t = jnp.sqrt(1. - b)
    x_pred = (z_t - sigma_t * eps_hat) / alpha_t

    return z_s, x_pred # return both if not jax.lax.fori_loop

def sample_fn(key, vdm, N_sample, T_sample, data_shape, sharding):
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
    x_sample = jax.vmap(data_generate_x, in_axes=(0, None, None))(
        z, vdm.gamma(0.), key
    )
    print("...sampled.")
    print("z, x_pred, x_sample", z.shape, x_pred.shape, x_sample.shape)
    return z, x_pred, x_sample

def plot_samples(samples, filename):
    n_side = int(math.sqrt(len(samples))) 
    fig, axs = plt.subplots(n_side, n_side, dpi=300, figsize=(8., 8.))
    c = 0
    for i in range(n_side):
        for j in range(n_side):
            ax = axs[i, j]
            ax.imshow(samples[c].transpose(1, 2, 0), cmap=cmap)
            ax.axis("off")
            c += 1
    plt.subplots_adjust(wspace=0.01, hspace=0.01) 
    plt.savefig(os.path.join(imgs_dir, filename), bbox_inches="tight")
    plt.close()

def image_caster(images):
    return images / 2. + 0.5

def image_shaper(images):
    n, c, h, w = images.shape
    b = int(np.sqrt(n))
    return einops.rearrange(
        images, 
        "(b1 b2) c h w -> (b2 h) (b1 w) c", 
        b1=b,
        b2=b,
        h=h, 
        w=w, 
        c=c
    )

if __name__ == "__main__":
    key = jr.PRNGKey(0)

    # Data hyper-parameters
    dataset_name = "CIFAR10"
    context_dim = None
    data_shape = (1, 32, 32) if dataset_name == "EMNIST" else (3, 32, 32)
    # Model hyper-parameters
    model_name = "vdm_" + dataset_name
    init_gamma_0 = -13.3
    init_gamma_1 = 5. 
    activation = jax.nn.tanh
    T_train = 0 
    T_sample = 1000
    n_sample = 36
    # Optimization hyper-parameters
    n_epochs = 1000
    n_batch = 128 #256
    learning_rate = 5e-5
    # Plotting
    proj_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/little_studies/fishing/vdm/"
    imgs_dir = os.path.join(proj_dir, "imgs_" + dataset_name)
    cmap = "gnuplot"

    key_s, key_n = jr.split(key)
    score_network = ScoreNetwork(
        data_shape,
        context_dim,
        init_gamma_0, 
        init_gamma_1, 
        key=key_s
    )
    noise_schedule = NoiseScheduleNN(
        init_gamma_0, init_gamma_1, key=key_n
    )
    vdm = VDM(score_network, noise_schedule)

    vdm = eqx.tree_deserialise_leaves(model_name, vdm)
    print("Loaded:", model_name)

    for i in range(5):
        key = jr.fold_in(key, i)

        zs, x_preds, samples = sample_fn(
            key, vdm, n_sample, T_sample, data_shape, sharding=None
        )

        samples = image_shaper(image_caster(samples))
        zs = image_shaper(image_caster(zs))
        x_preds = image_shaper(image_caster(x_preds))
        # plot_samples(samples, f"inference_x_{i}.png")
        # plot_samples(zs, f"inference_z_{i}.png")

        fig, axs = plt.subplots(1, 2, figsize=(16., 8.), dpi=300) 
        ax = axs[0]
        ax.imshow(zs)
        ax.axis("off")
        ax = axs[1]
        ax.imshow(samples)
        ax.axis("off")
        # ax = axs[2]
        # ax.imshow(x_preds)
        # ax.axis("off")
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig(
            os.path.join(imgs_dir, f"inferences_{i}.png"), 
            bbox_inches="tight"
        )
        plt.close()
