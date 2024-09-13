import os
import jax.random as jr 
from jaxtyping import Key
from torchvision import transforms, datasets

from .utils import ScalerDataset, Scaler, _TorchDataLoader


def mnist(key: Key, data_dir: str) -> ScalerDataset:
    key_train, key_valid = jr.split(key)
    n_pix = 32 # Force power of 2 resolution
    data_shape = (1, n_pix, n_pix)
    context_shape = (1,)

    scaler = Scaler(x_min=0., x_max=1.)

    train_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.ToTensor(), # This magically [0,255] -> [0,1]??
            transforms.Lambda(scaler.forward) # [0,1] -> [-1,1]
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.ToTensor(),
            transforms.Lambda(scaler.forward)
        ]
    )
    train_dataset = datasets.MNIST(
        os.path.join(data_dir, "mnist/"), 
        train=True,
        download=True, 
        transform=train_transform
    )
    valid_dataset = datasets.MNIST(
        os.path.join(data_dir, "mnist/"), 
        train=False, 
        download=True, 
        transform=valid_transform
    )

    train_dataloader = _TorchDataLoader(train_dataset, key=key_train)
    valid_dataloader = _TorchDataLoader(valid_dataset, key=key_valid)
    return ScalerDataset(
        name="mnist",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        scaler=scaler
    )