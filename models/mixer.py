import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np 
import einops


class MixerBlock(eqx.Module):
    patch_mixer: eqx.nn.MLP
    hidden_mixer: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key):
        tkey, ckey = jr.split(key)
        self.patch_mixer = eqx.nn.MLP(
            num_patches, num_patches, mix_patch_size, depth=1, key=tkey)
        self.hidden_mixer = eqx.nn.MLP(
            hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey)
        self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))

    def __call__(self, y):
        y = y + jax.vmap(self.patch_mixer)(self.norm1(y))
        y = einops.rearrange(y, "c p -> p c")
        y = y + jax.vmap(self.hidden_mixer)(self.norm2(y))
        y = einops.rearrange(y, "p c -> c p")
        return y


class Mixer2d(eqx.Module):
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.ConvTranspose2d
    blocks: list
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        img_size,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        context_dim,
        *,
        key):
        input_size, height, width = img_size
        assert (height % patch_size) == 0
        assert (width % patch_size) == 0
        num_patches = (height // patch_size) * (width // patch_size)
        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        self.conv_in = eqx.nn.Conv2d(
            input_size + 1, # Condition is tiled along with time as channels
            hidden_size, 
            patch_size, 
            stride=patch_size, 
            key=inkey)
        self.conv_out = eqx.nn.ConvTranspose2d(
            hidden_size, 
            input_size, 
            patch_size, 
            stride=patch_size, 
            key=outkey)
        self.blocks = [
            MixerBlock(
                num_patches, 
                hidden_size, 
                mix_patch_size, 
                mix_hidden_size, 
                key=bkey) 
            for bkey in bkeys]
        self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))

    def __call__(self, t, y):
        _, height, width = y.shape
        t = jnp.atleast_1d(t)
        _t = einops.repeat(
            t, "1 -> 1 h w", h=height, w=width)
        y = jnp.concatenate([y, _t])
        y = self.conv_in(y)
        _, patch_height, patch_width = y.shape
        y = einops.rearrange(
            y, "c h w -> c (h w)")
        for block in self.blocks:
            y = block(y)
        y = self.norm(y)
        y = einops.rearrange(
            y, "c (h w) -> c h w", h=patch_height, w=patch_width)
        return self.conv_out(y)


def trunc_init(weight: jax.Array, key: jr.PRNGKey) -> jax.Array:
    _, in_, *_ = weight.shape
    stddev = jnp.sqrt(1. / in_)
    return stddev * jr.truncated_normal(key, lower=-2., upper=2.)

def init_linear_weight(model, init_fn, key):
    layer_types = (eqx.nn.Linear, eqx.nn.Conv2d, eqx.nn.ConvTranspose2d)
    is_weighted = lambda x: isinstance(x, layer_types)
    get_weights = lambda m: [
        x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_weighted) if is_weighted(x)]
    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jr.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model