import jax.random as jr 
from jaxtyping import Key
from sklearn.datasets import make_moons

from .utils import ScalerDataset, Scaler, _InMemoryDataLoader


def cifar10(key: Key) -> ScalerDataset:
    key_train, key_valid = jr.split(key)
    data_shape = (2,)
    context_shape = (1,)

    scaler = Scaler(x_min=0., x_max=1.)

    Xt, Yt = make_moons(10_000, noise=0.05)
    Xv, Yv = make_moons(10_000, noise=0.05)

    train_dataloader = _InMemoryDataLoader(Xt, Yt, key=key_train)
    valid_dataloader = _InMemoryDataLoader(Xv, Yv, key=key_valid)
    return ScalerDataset(
        name="moons",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        scaler=scaler
    )