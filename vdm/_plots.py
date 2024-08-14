import os
import math
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_metrics(vdm, losses, metrics, filename):
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
    plt.savefig(filename)
    plt.close()

    # Plot the learned noise schedule, gamma = gamma(t \in [0., 1.]) 
    print('gamma_0', vdm.gamma(0.))
    print('gamma_1', vdm.gamma(1.))


def plot_train_sample(dataset, sample_size, vs=None, cmap="Blues", filename=None):
    def imgs_to_grid(X):
        """ Arrange images to one grid image """
        # Assumes square number of imgs
        N, c, h, w = X.shape
        n = int(np.sqrt(N))
        X_grid = rearrange(
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


def plot_samples(samples, filename, cmap="Blues"):
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
    plt.savefig(filename, bbox_inches="tight")
    plt.close()