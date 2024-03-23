import jax
import ml_collections

def d(**kwargs):
  """Helper of creating a config dict."""
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    """Get the hyperparameters for the model"""
    config = ml_collections.ConfigDict()
    config.exp_name = "VDM_CIFAR10"
    config.model_type = "VDM"
    config.ckpt_restore_dir = 'None'

    config.data = d(
        dataset='CIFAR10',  
        data_shape=(3, 32, 32),
        standardisation=(0.5, 0.5, 0.5)
    )

    config.model = d(
        # Configurations of the noise schedule
        gamma_type='learnable_scalar',  # learnable_scalar / learnable_nnet / fixed
        init_gamma_0=-13.3,
        init_gamma_1=5.,
        # Architecture
        activation=jax.nn.tanh,
        # Configurations of the score model

    )

    config.training = d(
        seed=0,
        n_batch=128,
        n_epochs=1_000,
        T_train=0
    )
    config.sampling = d(
       T_sample=1000,
       n_sample=36
    )
    config.optimizer = d(
        name='adamw',
        args=d(
            b1=0.9,
            b2=0.99,
            eps=1e-8,
            weight_decay=0.01,
        ),
        learning_rate=5e-5,
        lr_decay=False,
        ema_rate=0.9999,
    )

    return config