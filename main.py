import time
import math
import os
import jax
import jax.numpy as jnp
import jax.random as jr 
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import equinox as eqx
import optax
import numpy as np
import einops
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange
from tensorflow_probability.substrates.jax import distributions as tfd

from models import VDM, NoiseScheduleNN, Mixer2d, UNet
from data.utils import ScalerDataset


def get_sharding():
    n_devices = len(jax.local_devices())
    print(f"Running on {n_devices} devices: \n{jax.local_devices()}")

    use_sharding = n_devices > 1

    # Sharding mesh: speed and allow training on high resolution?
    if use_sharding:
        # Split array evenly across data dimensions, this reshapes automatically
        mesh = Mesh(jax.devices(), ('x',))
        sharding = NamedSharding(mesh, P('x'))
        print(f"Sharding:\n {sharding}")
    else:
        sharding = None
    return sharding


def put_on_sharding(x, sharding):
    if sharding is not None:
        x = jax.device_put(x, sharding)
    return x


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


def _var(gamma):
    return jax.nn.sigmoid(gamma)


def _alpha_sigma(gamma):
    var = _var(gamma)
    return jnp.sqrt(1. - var), jnp.sqrt(var)


def _encode(x):
    return x#(x - x.mean()) / x.std()


def _decode(z_0_rescaled, gamma_0, key):
    dist = tfd.Independent(
        tfd.Normal(
            z_0_rescaled, jnp.array([1e-3])
        ),  #jnp.exp(0.5 * gamma_0)),
        reinterpreted_batch_ndims=3
    )
    return dist.sample(seed=key)


def _logprob(x, z_0_rescaled, gamma_0):
    dist = tfd.Independent(
        tfd.Normal(
            z_0_rescaled, jnp.array([1e-3])
        ),  #jnp.exp(0.5 * gamma_0)),
        reinterpreted_batch_ndims=3
    )
    return dist.log_prob(x)


def _generate_x(z_0, gamma_0, key):
    alpha_0, _ = _alpha_sigma(gamma_0)
    z_0_rescaled = z_0 / alpha_0
    sample = _decode(z_0_rescaled, gamma_0, key)
    return sample

def get_gamma_alpha_sigma(vdm, t):
    gamma_t = vdm.gamma(t)
    return gamma_t, *_alpha_sigma(gamma_t)

def vlb(vdm, x, key, t=None, shard=jax.sharding.Sharding | None):
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
    if T_train > 0:
        t = jnp.ceil(t * T_train) / T_train

    # sample z_t 
    eps = jr.normal(key_d, shape=f.shape)
    eps = jax.device_put(eps, shard)
    z_t = alpha_t * f + sigma_t * eps
    # compute predicted noise
    key, _ = jr.split(key)
    eps_ = vdm.score(z_t, gamma_t, key=key)
    # compute MSE of predicted noise
    loss_diff_mse = jnp.square(eps - eps_).sum()

    if T_train == 0:
        # loss for infinite depth T, i.e. continuous time
        _, g_t_grad = jax.jvp(vdm.gamma, (t,), (jnp.ones_like(t),))
        loss_diff = 0.5 * g_t_grad * loss_diff_mse
    else:
        # loss for finite depth T, i.e. discrete time
        s = t - (1. / T_train)
        gamma_s = vdm.gamma(s)
        loss_diff = 0.5 * T_train * jnp.expm1(gamma_t - gamma_s) * loss_diff_mse

    loss = loss_klz + loss_recon + loss_diff
    metrics = [loss_klz, loss_recon, loss_diff]
    return loss, metrics


def loss_fn(vdm, key, x, t, shard):
    loss, metrics = vlb(vdm, x, key, t, shard)
    return loss, metrics


@eqx.filter_jit
def batch_loss_fn(vdm, key, x, shard): 
    key, key_t = jr.split(key)
    keys = jr.split(key, len(x))
    # Antithetic time sampling for lower variance VLB(x)
    n = len(x)
    t = jr.uniform(key_t, (n,), minval=0., maxval=1. / n)
    t = t + (1. / n) * jnp.arange(n)
    _fn = eqx.filter_vmap(loss_fn, in_axes=(None, 0, 0, 0, None))
    loss, metrics = _fn(vdm, keys, x, t, shard)
    return loss.mean(), [m.mean() for m in metrics]


@eqx.filter_jit
def make_step(vdm, x, key, opt_state, opt_update, shard):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn, has_aux=True)
    (loss, metrics), grads = loss_fn(vdm, key, x, shard)
    updates, opt_state = opt_update(grads, opt_state, vdm)
    vdm = eqx.apply_updates(vdm, updates)
    return vdm, loss, metrics, opt_state


def unbatch(batch, sharding=None):
    x, y = batch
    if sharding is not None:
        x, y = jax.device_put((x, y), sharding)
    return x, y


def infinite_trainloader(dataloader):
    while True:
        yield from dataloader 


def plot_metrics(losses, metrics):
    Lt, Lv = jnp.asarray(losses).T
    metrics_t, metrics_v = jnp.split(jnp.asarray(metrics), 2, axis=1)

    t_ = jnp.linspace(0., 1., 1_000)
    steps = range(len(metrics_t))
    plotter = lambda ax: ax.semilogy

    gammas = jax.vmap(vdm.gamma)(t_)
    SNRs = jnp.exp(-gammas)
    vars = jax.nn.sigmoid(gammas) 

    kl_t, recon_t, diffusion_t = metrics_t.squeeze().T
    kl_v, recon_v, diffusion_v = metrics_t.squeeze().T

    fig, axs = plt.subplots(2, 3, figsize=(10., 6.))
    ax = axs[0, 0]
    ax.set_title("L")
    ax.plot(steps, Lt, color="navy", label="train")
    ax.plot(steps, Lv, color="firebrick", label="valid") # plotter(ax)
    ax.legend()
    ax = axs[0, 1]
    ax.set_title("kl")
    plotter(ax)(steps, kl_t, color="forestgreen")
    plotter(ax)(steps, kl_v, color="forestgreen", linestyle=":")
    ax = axs[0, 2]
    ax.set_title("reconstruction")
    ax.plot(steps, recon_t, color="darkorange")
    ax.plot(steps, recon_v, color="darkorange", linestyle=":")
    ax = axs[1, 0]
    ax.set_title("diffusion")
    plotter(ax)(steps, diffusion_t, color="rebeccapurple")
    plotter(ax)(steps, diffusion_v, color="rebeccapurple", linestyle=":")
    ax = axs[1, 1]
    ax.set_title(r"$\alpha(t)$, $\sigma(t)$")
    ax.plot(t_, jnp.sqrt(vars), label=r"$\sigma(t)$")
    ax.plot(t_, jnp.sqrt(1. - vars), label=r"$\alpha(t)$")
    ax.legend()
    ax = axs[1, 2]
    ax.set_title(r"SNR$(t)$, $\gamma(t)$")
    ax.plot(t_, gammas, color="rebeccapurple", label=r"$\gamma(t)$")
    ax.legend()
    ax_ = ax.twinx()
    ax_.plot(t_, SNRs, color="forestgreen", label=r"SNR$(t)$")
    ax_.set_ylabel("$SNR$")
    ax_.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(imgs_dir, "loss.png"))
    plt.close()

    # Plot the learned noise schedule, gamma = gamma(t \in [0., 1.]) 
    print('gamma_0', vdm.gamma(0.))
    print('gamma_1', vdm.gamma(1.))


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
    x_sample = jax.vmap(_generate_x, in_axes=(0, None, None))(
        z, vdm.gamma(0.), key
    )
    print("...sampled.")
    print("z, x_pred, x_sample", z.shape, x_pred.shape, x_sample.shape)
    return z, x_pred, x_sample


def plot_train_sample(dataset: ScalerDataset, sample_size, vs=None, cmap=None, filename=None):
    def imgs_to_grid(X):
        """ Arrange images to one grid image """
        # Assumes square number of imgs
        N, c, h, w = X.shape
        n = int(np.sqrt(N))
        X_grid = einops.rearrange(
            X, 
            "(n1 n2) c h w -> (n1 h) (n2 w) c", 
            c=c,
            n1=n, 
            n2=n, 
        )
        return X_grid

    def _add_spacing(img, img_size):
        """ Add whitespace between images on a grid """
        # Assuming channels added from `imgs_to_grid`, and square imgs
        h, w, c = img.shape
        idx = jnp.arange(img_size, h, img_size)
        # NaNs not registered by colormaps?
        img_  = jnp.insert(img, idx, jnp.nan, axis=0)
        img_  = jnp.insert(img_, idx, jnp.nan, axis=1)
        return img_

    def X_onto_ax(_X, fig, ax, vs, cmap):
        """ Drop a sample _X onto an ax by gridding it first """
        _, c, img_size, _ = _X.shape
        im = ax.imshow(
            _add_spacing(imgs_to_grid(_X), img_size), 
            # **vs, # 'vs' is dict of imshow vmin and vmax
            cmap=cmap
        )
        ax.axis("off")
        # If only one channel, use colorbar
        if c == 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
        else:
            pass

    # Unscale data from dataloader (would need to do this for Q also)
    X, Q = next(dataset.train_dataloader.loop(sample_size))

    print("batch X", X.min(), X.max())

    # Re-scale data to [0, 1] from [-1, 1] (out of loader) for plotting
    X = dataset.scaler.reverse(X)[:sample_size]
    print(X.min(), X.max())

    fig, ax = plt.subplots(dpi=300)
    if dataset.name != "moons":
        X_onto_ax(X, fig, ax, vs, cmap)
    else: 
        ax.scatter(*X.T, c=Q, cmap="PiYG")
    del X, Q
    ax.set_title("$\delta_g$")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


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


if __name__ == "__main__":
    from data.cifar10 import cifar10
    from data.emnist import emnist

    key = jr.PRNGKey(0)

    shard = get_sharding()

    data_path = "/project/ls-gruen/users/jed.homer/jakob_ift/data/" 
    dataset_name = "CIFAR10" #"EMNIST"

    dataset = cifar10(key)

    # Data hyper-parameters
    context_dim = None
    data_shape = dataset.data_shape
    # Model hyper-parameters
    model_name = "vdm_" + dataset_name
    init_gamma_0 = -13.3
    init_gamma_1 = 5. 
    T_train = 0 
    T_sample = 1000
    n_sample = 64 # Must be sharding congruent?
    # Optimization hyper-parameters
    n_steps = 1_000_000
    # n_batch = 256 #
    n_batch = 128 
    learning_rate = 5e-5
    # Plotting
    proj_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/vdm/"
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

    opt = optax.adamw(learning_rate=learning_rate) # schedule
    opt_state = opt.init(eqx.filter(vdm, eqx.is_inexact_array))
  
    plot_train_sample(dataset, 25, filename=os.path.join(imgs_dir, "data.png"))
    key, sample_key = jr.split(key)

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
                vdm, x, train_key, opt_state, opt.update, shard
            )

            # Validate
            x, y = unbatch(valid_batch, shard)

            loss_valid, valid_metrics = batch_loss_fn(vdm, valid_key, x, shard)

            # Record
            losses += [(loss_train, loss_valid)]
            metrics += [(train_metrics, valid_metrics)]
            steps.set_postfix(
                Lt=f"{losses[-1][0]:.4E}", Lv=f"{losses[-1][1]:.4E}"
            )

            # Sample
            if s % 1000 == 0:
                _, _, samples = sample_fn(
                    sample_key, vdm, n_sample, T_sample, data_shape, shard
                )
                samples = dataset.scaler.reverse(samples) # [-1, 1] -> [0, 1]

                plot_samples(samples, f"samples_{s:06d}.png")
                plot_metrics(losses, metrics)

                eqx.tree_serialise_leaves(model_name, vdm)

    def J_fn(vlb_fn, q, x, t, key):
        # Is product of gradients faster than Hessian? VLB ~ logL(x)
        # jax.hessian(vlb_fn)
        dL = jax.jacfwd(vlb_fn)(q, x, t, key)
        return jnp.matmul(dL.T, dL)

    def get_J_fn(vlb):
        return lambda q, x, t, key: vlb(vdm, x, key, t)

    @eqx.filter_jit
    def observed_information(key, q, sample_fn, J_fn, n_samples):
        # Get samples from model, at 'q' in parameter space
        key_sample, key_J = jr.split(key)
        keys = jr.split(key_sample, n_samples)
        # sample_fn = get_sample_fn(model, int_beta, data_shape, dt0, t1)
        x = jax.vmap(sample_fn, in_axes=(0, None))(keys, q)
        # Calculate observed Fisher information under model (args ordered for argnum=0)
        keys = jr.split(key_J, n_samples)
        # L_x_q = jax.vmap(log_likelihood_fn, in_axes=(None, 0, 0))(q, x, keys)
        Js = jax.vmap(J_fn, in_axes=(None, 0, 0))(q, x, keys)
        F = Js.mean(axis=0)
        return F

    def get_observed_information_fn(sample_fn, log_likelihood_fn, n_samples):
        return eqx.filter_jit(
            lambda key, q: observed_information(
                key, q, sample_fn, log_likelihood_fn, n_samples
            )
        )

    def logdet(F):
        a, b = jnp.linalg.slogdet(F)
        return a * b

"""

def generate(key, vdm, data_shape):
    @jax.jit
    def sample_step(key, i, T, z_t):
        key = jr.fold_in(key, i)
        eps = jr.normal(key, z_t.shape)
        
        t = (T - i) / T 
        s = (T - i - 1) / T

        g_s = vdm.gamma(s)
        g_t = vdm.gamma(t)

        eps_hat = vdm.score_network(z_t, g_t)
        
        a = jax.nn.sigmoid(g_s)
        b = jax.nn.sigmoid(g_t)
        c = -jnp.expm1(g_t - g_s)
        sigma_t = jnp.sqrt(_var(g_t))
        z_s = jnp.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) \
            + jnp.sqrt((1. - a) * c) * eps
        return z_s

    # first generate latent
    key, key_eps = jr.split(key)

    # Prior sample
    z_T = jr.normal(key_eps, data_shape)

    def body_fn(i, z_t):
        return sample_step(key, i, T_sample, z_t)

    # Last latent sample (given zt above)
    z_0 = jax.lax.fori_loop(
        lower=0, upper=T_sample, body_fun=body_fn, init_val=z_T
    )

    gamma_0 = vdm.gamma(0.0)
    var0 = _var(gamma_0)
    z_0_rescaled = z_0 / np.sqrt(1. - var0)
    # return z_0_rescaled #vdm.apply(params, z0_rescaled, conditioning, method=vdm.decode)
    return decode(z_0_rescaled, gamma_0, key)

def sample_fn(key, vdm, N_sample, T_sample, data_shape, sharding):
    print("Sampling...")
    if 0:
        # sample z_T from the diffusion model
        key, _ = jr.split(key)
        z_T = jr.normal(key, (N_sample,) + data_shape)

        if sharding is not None:
            z_T = jax.device_put(z_T, sharding)

        z = [z_T]

        x_pred = []
        for i in trange(T_sample):
            key, _ = jr.split(key)
            _z, _x_pred = sample_step(i, vdm, T_sample, z[-1], key, sharding)
            z.append(_z)
            x_pred.append(_x_pred)
    else:
        z = jr.normal(key, (N_sample,) + data_shape)
        if sharding is not None:
            z = jax.device_put(z, sharding)

        # def body_fn(i, z_t_x):
        #     z_t, _ = z_t_x
        #     return sample_step(i, vdm, T_sample, z_t, key)
        def body_fn(i, z_t):
            fn = lambda z_t, i: sample_step(
                i, vdm, T_sample, z_t, key
            )
            return fn(z_t, i) # z_t must be first argument

        z = jax.lax.fori_loop(
            lower=0, upper=T_sample, body_fun=body_fn, init_val=z 
        )
    key, _ = jr.split(key)
    x_sample = data_generate_x(z[-1], vdm.gamma(0.), key)
    print("...sampled.")
    return jnp.asarray(z), jnp.asarray(x_pred), jnp.asarray(x_sample)
"""
# opt = optax.adamw(learning_rate)
# opt = optax.chain(
#     optax.clip_by_global_norm(1.0),  
#     optax.scale_by_adam(),  
#     # optax.scale_by_schedule(scheduler),  
#     optax.scale(-1.0)
# )

# schedule = optax.warmup_cosine_decay_schedule(
#     init_value=0.0,
#     peak_value=learning_rate,
#     warmup_steps=50,
#     decay_steps=100,
#     end_value=1e-5,
# )

# opt = optax.chain(
#     # optax.clip_by_global_norm(1.0),  
#     optax.adamw(learning_rate=learning_rate), # schedule
# )