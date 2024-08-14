import os
import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import einops
import numpy as np
import matplotlib.pyplot as plt

from models import VDM, NoiseScheduleNN, ScoreNetwork
from sample import sample_fn
from plots import plot_samples
from utils import get_sharding


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
    from data.cifar10 import cifar10

    key = jr.PRNGKey(0)

    dataset = cifar10(key)

    sharding = get_sharding()

    # Data hyper-parameters
    context_dim = None
    data_shape = dataset.data_shape
    dataset_name = dataset.name
    # Model hyper-parameters
    model_name = "vdm_" + dataset_name
    init_gamma_0 = -13.3
    init_gamma_1 = 5. 
    activation = jax.nn.tanh
    T_train = 0 
    T_sample = 1000
    n_sample = 64
    # Plotting
    proj_dir = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/vdm/"
    imgs_dir = os.path.join(proj_dir, "imgs_" + dataset_name)

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
        print("inference ", i)
        key = jr.fold_in(key, i)

        zs, x_preds, samples = sample_fn(
            key, vdm, n_sample, T_sample, data_shape, sharding=sharding
        )
        print("sampled", samples.min(), samples.max())

        samples = image_shaper(dataset.scaler.reverse(samples))
        zs = image_shaper(dataset.scaler.reverse(zs))
        x_preds = image_shaper(dataset.scaler.reverse(x_preds))

        print("scaled", samples.min(), samples.max())

        fig, axs = plt.subplots(1, 2, figsize=(16., 8.), dpi=300) 
        ax = axs[0]
        ax.imshow(zs)
        ax.axis("off")
        ax = axs[1]
        ax.imshow(samples)
        ax.axis("off")
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig(
            os.path.join(imgs_dir, f"inferences_{i}.png"), 
            bbox_inches="tight"
        )
        plt.close()