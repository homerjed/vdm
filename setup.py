from setuptools import find_packages, setup

setup(
    name="vdm",
    packages=find_packages(where="vdm"),
    package_dir={"" : "vdm"},
    url="https://github.com/homerjed/vdm/",
    author="Jed Homer",
    author_email="jedhmr@gmail.com",
    license="MIT",
    keywords=[
        "artificial intelligence",
        "machine learning",
        "diffusion",
        "variational diffusion model",
        "score based diffusion",
        "generative models"
    ],
    install_requires=[
        "jax",
        "equinox",
        "diffrax",
        "optax",
        "einops",
        "numpy",
        "matplotlib",
        "torch",
        "torchvision",
        "tqdm",
    ],
)