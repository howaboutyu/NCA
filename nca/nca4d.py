"""A four-spatial-dimensional neural cellular automaton.

The state layout is ``N,C,Z,Y,X,A``.  Channels 0:3 are RGB, channel 3 is
alpha/liveness, and the remaining channels are hidden state.  Rollout steps
remain the automaton's time axis; the extra ``A`` axis is spatial.
"""

from itertools import product
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


class Conv4D(nn.Module):
    """JAX-native dense Conv4D that bypasses cuDNN's three-dimension limit.

    The implementation expands a small SAME-padded kernel into static spatial
    offsets and contracts each offset with an einsum. This is slower than a
    vendor convolution, but it remains GPU-compatible for the small kernels
    used by this experiment and preserves a true learned 4D receptive field.
    """

    features: int
    kernel_size: tuple[int, int, int, int]
    kernel_init: Callable = nn.initializers.glorot_uniform()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        if inputs.ndim != 6:
            raise ValueError("Conv4D expects an N,D,H,W,A,C input")
        if any(size % 2 == 0 for size in self.kernel_size):
            raise ValueError("Conv4D currently requires odd kernel dimensions")
        in_channels = inputs.shape[-1]
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (*self.kernel_size, in_channels, self.features),
        )
        bias = self.param("bias", self.bias_init, (self.features,))
        padding = tuple((size // 2, size // 2) for size in self.kernel_size)
        padded = jnp.pad(
            inputs,
            ((0, 0), padding[0], padding[1], padding[2], padding[3], (0, 0)),
        )
        output = jnp.zeros(inputs.shape[:-1] + (self.features,), dtype=inputs.dtype)
        spatial_shape = inputs.shape[1:-1]
        for offset in product(*(range(size) for size in self.kernel_size)):
            slices = tuple(
                slice(index, index + length)
                for index, length in zip(offset, spatial_shape)
            )
            patch = padded[(slice(None), *slices, slice(None))]
            output = output + jnp.einsum(
                "bdhwac,co->bdhwao", patch, kernel[offset]
            )
        return output + bias


class UpdateModel4D(nn.Module):
    """Conditional local Conv4D update rule."""

    state_channels: int = 16
    hidden_channels: int = 64
    pokemon_vocab_size: int = 0
    pokemon_embedding_dim: int = 32
    kernel_init: Callable = nn.initializers.glorot_uniform

    def setup(self) -> None:
        self.perception = Conv4D(
            features=3 * self.state_channels,
            kernel_size=(3, 3, 3, 3),
            kernel_init=self.kernel_init(),
        )
        if self.pokemon_vocab_size > 0:
            self.pokemon_embedding = nn.Embed(
                num_embeddings=self.pokemon_vocab_size,
                features=self.pokemon_embedding_dim,
            )
        self.conv_1 = Conv4D(
            features=self.hidden_channels,
            kernel_size=(1, 1, 1, 1),
            kernel_init=self.kernel_init(),
        )
        # Zero initialization keeps the seed unchanged at the start of a run.
        self.conv_2 = Conv4D(
            features=self.state_channels,
            kernel_size=(1, 1, 1, 1),
            kernel_init=nn.initializers.zeros,
        )

    def __call__(
        self, state_ndhwac: jax.Array, pokemon_ids: jax.Array | None = None
    ) -> jax.Array:
        """Predict one local update from an ``N,D,H,W,A,C`` state."""
        x = self.perception(state_ndhwac)
        if self.pokemon_vocab_size > 0:
            if pokemon_ids is None:
                raise ValueError("pokemon_ids are required for conditional 4D NCA")
            embedding = self.pokemon_embedding(pokemon_ids)
            embedding = embedding[:, None, None, None, None, :]
            x = jnp.concatenate(
                [x, jnp.broadcast_to(embedding, x.shape[:-1] + (embedding.shape[-1],))],
                axis=-1,
            )
        x = nn.relu(self.conv_1(x))
        return self.conv_2(x)


def alive_mask_4d(alpha: jax.Array, threshold: float = 0.1) -> jax.Array:
    """Keep cells alive when alpha reaches a 3×3×3×3 neighbourhood."""
    pooled = jax.lax.reduce_window(
        alpha,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 3, 3, 3, 3),
        window_strides=(1, 1, 1, 1, 1),
        padding="SAME",
    )
    return pooled > threshold


def cell_update_4d(
    key: jax.Array,
    state_ncdhwa: jax.Array,
    model: UpdateModel4D,
    params: object,
    update_probability: float = 0.5,
    state_clip: float = 16.0,
    pokemon_ids: jax.Array | None = None,
) -> jax.Array:
    """Apply one stochastic Conv4D update to an ``N,C,Z,Y,X,A`` state."""
    if state_ncdhwa.ndim != 6 or state_ncdhwa.shape[1] < 4:
        raise ValueError("4D NCA state must have shape N,C,Z,Y,X,A with at least 4 channels")
    pre_alive = alive_mask_4d(state_ncdhwa[:, 3])
    state_ndhwac = jnp.transpose(state_ncdhwa, (0, 2, 3, 4, 5, 1))
    delta = model.apply(params, state_ndhwac, pokemon_ids=pokemon_ids)
    update_mask = jax.random.uniform(key, delta.shape[:-1]) < update_probability
    delta_ncdhwa = jnp.transpose(delta * update_mask[..., None], (0, 5, 1, 2, 3, 4))
    next_state = state_ncdhwa + delta_ncdhwa
    next_state = jnp.nan_to_num(
        next_state, nan=0.0, posinf=state_clip, neginf=-state_clip
    )
    next_state = jnp.clip(next_state, -state_clip, state_clip)
    post_alive = alive_mask_4d(next_state[:, 3])
    return next_state * (pre_alive & post_alive)[:, None]


def rollout_4d(
    key: jax.Array,
    state_ncdhwa: jax.Array,
    model: UpdateModel4D,
    params: object,
    steps: int,
    update_probability: float = 0.5,
    state_clip: float = 16.0,
    pokemon_ids: jax.Array | None = None,
) -> jax.Array:
    """Run a fixed number of Conv4D updates; rollout time stays separate from A."""

    def rematerialized_update(current_state: jax.Array, step_key: jax.Array) -> jax.Array:
        return cell_update_4d(
            step_key,
            current_state,
            model,
            params,
            update_probability,
            state_clip,
            pokemon_ids,
        )

    update_step = jax.checkpoint(rematerialized_update)
    step_keys = jax.random.split(key, steps)

    def scan_step(current_state: jax.Array, step_key: jax.Array):
        return update_step(current_state, step_key), None

    state_ncdhwa, _ = jax.lax.scan(scan_step, state_ncdhwa, step_keys)
    return state_ncdhwa


def rollout_4d_frames(
    key: jax.Array,
    state_ncdhwa: jax.Array,
    model: UpdateModel4D,
    params: object,
    steps: int,
    update_probability: float = 0.5,
    state_clip: float = 16.0,
    pokemon_ids: jax.Array | None = None,
) -> jax.Array:
    """Return every state in a Conv4D rollout for evaluation GIFs."""

    def scan_step(current_state: jax.Array, step_key: jax.Array):
        next_state = cell_update_4d(
            step_key,
            current_state,
            model,
            params,
            update_probability,
            state_clip,
            pokemon_ids,
        )
        return next_state, next_state

    step_keys = jax.random.split(key, steps)
    _, frames = jax.lax.scan(scan_step, state_ncdhwa, step_keys)
    return frames


def render_xy_plane_4d(
    state_ncdhwa: jax.Array, z_index: int, a_index: int
) -> jax.Array:
    """Extract one ordinary RGBA ``Y,X`` plane at a fixed ``Z,A`` coordinate."""
    return jnp.clip(state_ncdhwa[:, :4, z_index, :, :, a_index], 0.0, 1.0)


def project_xy_over_a_4d(state_ncdhwa: jax.Array, z_index: int) -> jax.Array:
    """Alpha-composite all A slices into one RGBA ``Y,X`` reconstruction.

    The composition order is increasing A.  This gives every A slice a path
    to the 2D reconstruction loss instead of supervising only one hyperplane.
    """
    rgba = jnp.clip(state_ncdhwa[:, :4, z_index], 0.0, 1.0)
    rgb = jnp.moveaxis(rgba[:, :3], 1, -1)  # N,Y,X,A,3
    alpha = rgba[:, 3]  # N,Y,X,A
    transmittance_before = jnp.concatenate(
        [jnp.ones_like(alpha[..., :1]), jnp.cumprod(1.0 - alpha[..., :-1], axis=-1)],
        axis=-1,
    )
    weights = alpha * transmittance_before
    composed_rgb = jnp.sum(rgb * weights[..., None], axis=-2)
    composed_alpha = 1.0 - jnp.prod(1.0 - alpha, axis=-1)
    return jnp.concatenate(
        [jnp.moveaxis(composed_rgb, -1, 1), composed_alpha[:, None]], axis=1
    )


def seed_volume_4d(
    batch_size: int,
    state_channels: int,
    depth: int,
    height: int,
    width: int,
    axis_size: int,
    seed_depth_index: int = 0,
    seed_axis_index: int = 0,
    random_seed_density: float = 0.0,
    random_seed_key: jax.Array | None = None,
) -> jax.Array:
    """Create a compact XY seed at one ``(Z,A)`` coordinate."""
    if state_channels < 4:
        raise ValueError("A visual 4D NCA requires RGB plus alpha")
    if not 0 <= seed_depth_index < depth:
        raise ValueError("seed_depth_index must be inside the volume")
    if not 0 <= seed_axis_index < axis_size:
        raise ValueError("seed_axis_index must be inside the A axis")
    state = jnp.zeros((batch_size, state_channels, depth, height, width, axis_size))
    y, x = height // 2, width // 2
    state = state.at[:, 3:, seed_depth_index, y - 1 : y + 2, x - 1 : x + 2, seed_axis_index].set(1.0)
    if random_seed_density > 0.0:
        if random_seed_key is None:
            random_seed_key = jax.random.PRNGKey(0)
        random_mask = jax.random.uniform(
            random_seed_key, (batch_size, depth, height, width, axis_size)
        ) < random_seed_density
        state = state.at[:, 3:].set(
            jnp.where(random_mask[:, None], 1.0, state[:, 3:])
        )
    return state
