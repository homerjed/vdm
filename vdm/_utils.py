import jax
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import equinox as eqx


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
    if sharding is not None:
        batch = eqx.filter_shard(batch, sharding)
    return batch


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