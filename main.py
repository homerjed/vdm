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

from models import VDM, NoiseScheduleNN, ScoreNetwork
from data.utils import ScalerDataset
from train import make_step, batch_loss_fn
from sample import sample_fn


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


def unbatch(batch, sharding=None):
    x, y = batch
    if sharding is not None:
        x, y = jax.device_put((x, y), sharding)
    return x, y


def plot_metrics(vdm, losses, metrics):
    Lt, Lv = jnp.asarray(losses).T
    metrics_t, metrics_v = jnp.split(jnp.asarray(metrics), 2, axis=1)

    t_ = jnp.linspace(0., 1., 1_000)
    steps = range(len(metrics_t))
    plotter = lambda ax: ax.semilogy

    gammas = jax.vmap(vdm.gamma)(t_)
    SNRs = jnp.exp(-gammas)
    vars = jax.nn.sigmoid(gammas) 

    kl_t, recon_t, diffusion_t = metrics_t.squeeze().T
    kl_v, recon_v, diffusion_v = metrics_v.squeeze().T

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


def plot_train_sample(dataset: ScalerDataset, sample_size, vs=None, cmap="Blues", filename=None):
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
    print("batch X (reverse scaled)", X.min(), X.max())

    fig, ax = plt.subplots(dpi=300)
    if dataset.name != "moons":
        X_onto_ax(X, fig, ax, vs, cmap)
    else: 
        ax.scatter(*X.T, c=Q, cmap="PiYG")
    del X, Q
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_samples(samples, filename):
    cmap = "Blues"
    n_side = int(math.sqrt(len(samples))) 
    samples = jnp.clip(samples, 0., 1.)
    # samples = jnp.clip(samples, 0., 1.)
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


def save_opt_state_and_model(
    opt_state, model, filename_opt, filename_model
):
    eqx.tree_serialise_leaves(filename_opt, opt_state)
    eqx.tree_serialise_leaves(filename_model, model)


def load_opt_state_and_model(
    opt_state, model, filename_opt, filename_model
):
    opt_state = eqx.tree_deserialise_leaves(filename_opt, opt_state)
    model = eqx.tree_deserialise_leaves(filename_model, model)
    return opt_state, model


if __name__ == "__main__":
    from data.cifar10 import cifar10
    from data.mnist import mnist

    key = jr.PRNGKey(0)
    key, model_key, sample_key = jr.split(key, 3)

    shard = get_sharding()

    data_path = "/project/ls-gruen/users/jed.homer/jakob_ift/data/" 

    # Data hyper-parameters
    dataset = mnist(key) # cifar10(key)
    context_dim = None
    data_shape = dataset.data_shape
    dataset_name = dataset.name
    # Model hyper-parameters
    model_name = "vdm_" + dataset_name
    init_gamma_0 = -13.3
    init_gamma_1 = 5. 
    T_sample = 1000
    n_sample = 64 # Must be sharding congruent?
    # Optimization hyper-parameters
    n_steps = 1_000_000
    n_batch = 256 
    learning_rate = 5e-5
    # Plotting
    proj_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/vdm/"
    imgs_dir = os.path.join(proj_dir, "imgs_" + dataset_name)
    cmap = "Blues"

    reload = False

    key_s, key_n = jr.split(model_key)
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
    vdm = VDM(
        score_network=score_network, 
        noise_network=noise_schedule
    )

    opt = optax.adamw(learning_rate=learning_rate)
    opt_state = opt.init(eqx.filter(vdm, eqx.is_inexact_array))

    if reload:
        opt_state, vdm = load_opt_state_and_model(
            opt_state=opt_state,
            vdm=vdm,
            filename_opt=model_name + "_opt.eqx",
            filename_model=model_name + ".eqx"
        )
  
    plot_train_sample(
        dataset, 
        sample_size=n_sample, 
        filename=os.path.join(imgs_dir, "data.png")
    )

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
                Lt=f"{losses[-1][0]:.3E}", 
                Lv=f"{losses[-1][1]:.3E}"
            )

            # Sample
            if s % 5_000 == 0:
                _, _, samples = sample_fn(
                    sample_key, vdm, n_sample, T_sample, data_shape, shard
                )
                samples = dataset.scaler.reverse(samples) # [-1, 1] -> [0, 1]

                plot_samples(samples, f"samples_{s:06d}.png")

                plot_metrics(vdm, losses, metrics)

                save_opt_state_and_model(
                    opt_state=opt_state,
                    model=vdm,
                    filename_opt=model_name + "_opt.eqx",
                    filename_model=model_name + ".eqx"
                )