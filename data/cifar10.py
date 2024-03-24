import abc
from typing import Tuple 
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as jr 
import numpy as np
import torch
from torchvision import transforms, datasets

from .utils import ScalerDataset, Scaler, _TorchDataLoader

def cifar10(key: jr.PRNGKey) -> ScalerDataset:
    key_train, key_valid = jr.split(key)
    n_pix = 32 # Native resolution for CIFAR10 
    data_shape = (3, n_pix, n_pix)
    context_shape = (1,)

    scaler = Scaler(x_min=0., x_max=1.)

    train_transform = transforms.Compose(
        [
            transforms.Resize((n_pix, n_pix)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # This magically does [0,255] -> [0,1]?
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
    train_dataset = datasets.CIFAR10(
        "datasets/" + "cifar10", 
        train=True, 
        download=True, 
        transform=train_transform
    )
    valid_dataset = datasets.CIFAR10(
        "datasets/" + "cifar10", 
        train=False, 
        download=True, 
        transform=valid_transform
    )

    train_dataloader = _TorchDataLoader(train_dataset, key=key_train)
    valid_dataloader = _TorchDataLoader(valid_dataset, key=key_valid)
    return ScalerDataset(
        name="cifar10",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape,
        context_shape=context_shape,
        scaler=scaler
    )