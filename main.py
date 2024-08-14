import os
import jax.random as jr 
import equinox as eqx
import optax

import vdm


if __name__ == "__main__":
    from data.cifar10 import cifar10
    from data.mnist import mnist

    key = jr.PRNGKey(0)
    key, model_key, train_key= jr.split(key, 3)

    shard = vdm.utils.get_sharding()

    data_path = "/project/ls-gruen/users/jed.homer/jakob_ift/data/" 

    # Data hyper-parameters
    dataset             = mnist(key) # cifar10(key)
    context_dim         = None
    data_shape          = dataset.data_shape
    dataset_name        = dataset.name
    # Model hyper-parameters
    model_name          = "vdm_" + dataset_name
    init_gamma_0        = -13.3
    init_gamma_1        = 5. 
    T_sample            = 1000
    n_sample            = 64 
    # Optimization hyper-parameters
    n_steps             = 1_000_000
    n_batch             = 256 
    learning_rate       = 5e-5
    # Plotting
    proj_dir            = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/vdm/"
    imgs_dir            = os.path.join(proj_dir, "imgs_" + dataset_name)

    reload = False # Reload model and optimiser state

    # Build VDM out of score network and noise schedule network
    key_score, key_noise = jr.split(model_key)
    score_network = vdm.models.ScoreNetwork(
        data_shape,
        context_dim,
        init_gamma_0, 
        init_gamma_1, 
        key=key_score
    )
    noise_schedule = vdm.models.NoiseScheduleNN(
        init_gamma_0, init_gamma_1, key=key_noise
    )
    vdm_model = vdm.models.VDM(
        score_network=score_network, 
        noise_network=noise_schedule
    )

    opt = optax.adamw(learning_rate=learning_rate)
    opt_state = opt.init(eqx.filter(vdm_model, eqx.is_inexact_array))

    # Reload if so desired
    if reload:
        opt_state, vdm_model = vdm.utils.load_opt_state_and_model(
            opt_state=opt_state,
            vdm=vdm_model,
            filename_opt=model_name + "_opt.eqx",
            filename_model=model_name + ".eqx"
        )
  
    # Plot training data for comparison
    vdm.plots.plot_train_sample(
        dataset, 
        sample_size=n_sample, 
        filename=os.path.join(imgs_dir, "data.png")
    )

    vdm = vdm.train.train(
        train_key, 
        vdm_model, 
        dataset, 
        opt, 
        opt_state, 
        n_steps=n_steps, 
        n_batch=n_batch, 
        n_sample=n_sample,
        T_sample=T_sample,
        shard=shard,
        model_name=model_name,
        imgs_dir=imgs_dir
    )