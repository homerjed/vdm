<h1 align='center'>vdm</h1>
<h2 align='center'>Variational Diffusion Models</h2>

Implementation and extension of [Variational Diffusion Models (Kingma++21)](https://arxiv.org/abs/2203.04176) in `jax` and `equinox`. 

### Synopsis 

A Variational Diffusion Model (VDM) is essentially an infinitely deep hierarchical model with an analytic encoding model for each of the latent variables. 

This design shares many similarities with a Variational Autoencoder (VAE) but unlike the VAE, the model is fit with three loss terms: the consistency (diffusion) loss, the reconstruction loss, and the prior KL-divergence.

Here training is implemented with the continuous-time depth consistency loss as opposed to a discretised SDE in the DDPM methods. 

### Features
* Conditional likelihood modelling,
* exotic score-network architectures (more to be added),
* multi-device training and inference.

### Usage

```
pip install variational-diffusion-models 
```

```
python main.py
``` 

See [examples](https://github.com/homerjed/vdm/tree/main/examples).

![alt text](https://github.com/homerjed/vdm/blob/master/figs/generated.png?raw=true)

#### CIFAR10 
![alt text](https://github.com/homerjed/vdm/blob/master/figs/cifar10.png?raw=true)

#### MNIST
![alt text](https://github.com/homerjed/vdm/blob/master/figs/mnist.png?raw=true)