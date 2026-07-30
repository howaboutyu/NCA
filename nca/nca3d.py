"""A cubic neural cellular automaton with a selected 2D RGBA face renderer.

The state is NCDHW. The standard experiment uses ``(16, 32, 32, 32)`` per
sample: channels 0:3 are RGB, channel 3 is alpha/liveness, and the remaining
channels are hidden state. Training compares RGBA from one selected spatial
face (for example ``state[:, :4, z=0]``) to the unchanged 2D target.
"""

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


def curl_divergence_3d(vector_field: jax.Array) -> jax.Array:
    """Return divergence and curl for an NDHWC three-component vector field.

    Spatial axes are ordered ``(z, y, x)`` and vector components are ordered
    ``(x, y, z)``.  Zero padding gives the fixed operator a non-periodic
    boundary condition consistent with the learned ``padding='SAME'``
    perception convolution.
    """
    if vector_field.shape[-1] != 3:
        raise ValueError("curl_divergence_3d expects exactly three vector components")
    padded = jnp.pad(vector_field, ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)))
    d_z = 0.5 * (padded[:, 2:, 1:-1, 1:-1] - padded[:, :-2, 1:-1, 1:-1])
    d_y = 0.5 * (padded[:, 1:-1, 2:, 1:-1] - padded[:, 1:-1, :-2, 1:-1])
    d_x = 0.5 * (padded[:, 1:-1, 1:-1, 2:] - padded[:, 1:-1, 1:-1, :-2])
    divergence = d_x[..., 0] + d_y[..., 1] + d_z[..., 2]
    curl_x = d_y[..., 2] - d_z[..., 1]
    curl_y = d_z[..., 0] - d_x[..., 2]
    curl_z = d_x[..., 1] - d_y[..., 0]
    return jnp.stack((divergence, curl_x, curl_y, curl_z), axis=-1)


class UpdateModel3D(nn.Module):
    """Conditional 3D NCA update rule with learned local Conv3D perception."""

    state_channels: int = 16
    hidden_channels: int = 96
    pokemon_vocab_size: int = 0
    pokemon_embedding_dim: int = 32
    use_curl_divergence: bool = False
    kernel_init: Callable = nn.initializers.glorot_uniform

    def setup(self) -> None:
        self.perception = nn.Conv(
            features=3 * self.state_channels,
            kernel_size=(3, 3, 3),
            padding="SAME",
            kernel_init=self.kernel_init(),
        )
        if self.pokemon_vocab_size > 0:
            self.pokemon_embedding = nn.Embed(
                num_embeddings=self.pokemon_vocab_size,
                features=self.pokemon_embedding_dim,
            )
        self.conv_1 = nn.Conv(
            features=self.hidden_channels,
            kernel_size=(1, 1, 1),
            padding="SAME",
            kernel_init=self.kernel_init(),
        )
        # Start as an identity automaton; updates are learned from the loss.
        self.conv_2 = nn.Conv(
            features=self.state_channels,
            kernel_size=(1, 1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )

    def __call__(
        self, state_ndhwc: jax.Array, pokemon_ids: jax.Array | None = None
    ) -> jax.Array:
        """Predict an update from learned local perception, optional operators, and ID."""
        x = self.perception(state_ndhwc)
        if self.use_curl_divergence:
            if state_ndhwc.shape[-1] < 7:
                raise ValueError("curl/divergence perception requires RGB, alpha, and 3 hidden channels")
            # Channels 4:7 are a dedicated learned vector field.  Keeping RGB
            # out of this operator avoids tying visual colour noise to physics
            # features while the learned Conv3D still sees every state channel.
            x = jnp.concatenate((x, curl_divergence_3d(state_ndhwc[..., 4:7])), axis=-1)
        if self.pokemon_vocab_size > 0:
            if pokemon_ids is None:
                raise ValueError("pokemon_ids are required for conditional 3D NCA")
            embedding = self.pokemon_embedding(pokemon_ids)
            embedding = embedding[:, None, None, None, :]
            x = jnp.concatenate(
                [x, jnp.broadcast_to(embedding, x.shape[:-1] + (embedding.shape[-1],))],
                axis=-1,
            )
        x = nn.relu(self.conv_1(x))
        return self.conv_2(x)


def alive_mask_3d(alpha: jax.Array, threshold: float = 0.1) -> jax.Array:
    """Keep cells alive when alpha reaches a 3×3×3 neighbourhood."""
    pooled = jax.lax.reduce_window(
        alpha,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 3, 3, 3),
        window_strides=(1, 1, 1, 1),
        padding="SAME",
    )
    return pooled > threshold


def cell_update_3d(
    key: jax.Array,
    state_ncdhw: jax.Array,
    model: UpdateModel3D,
    params: object,
    update_probability: float = 0.5,
    state_clip: float = 16.0,
    pokemon_ids: jax.Array | None = None,
) -> jax.Array:
    """Apply one stochastic 3D NCA update to an NCDHW state volume."""
    if state_ncdhw.shape[1] < 4:
        raise ValueError("3D NCA state requires RGB plus alpha (at least 4 channels)")
    pre_alive = alive_mask_3d(state_ncdhw[:, 3])
    delta = model.apply(
        params, jnp.transpose(state_ncdhw, (0, 2, 3, 4, 1)), pokemon_ids=pokemon_ids
    )
    update_mask = jax.random.uniform(key, delta.shape[:-1]) < update_probability
    next_state = state_ncdhw + jnp.transpose(
        delta * update_mask[..., None], (0, 4, 1, 2, 3)
    )
    next_state = jnp.nan_to_num(next_state, nan=0.0, posinf=state_clip, neginf=-state_clip)
    next_state = jnp.clip(next_state, -state_clip, state_clip)
    post_alive = alive_mask_3d(next_state[:, 3])
    return next_state * (pre_alive & post_alive)[:, None]


def rollout_3d(
    key: jax.Array,
    state_ncdhw: jax.Array,
    model: UpdateModel3D,
    params: object,
    steps: int,
    update_probability: float = 0.5,
    state_clip: float = 16.0,
    pokemon_ids: jax.Array | None = None,
) -> jax.Array:
    """Run a fixed number of volumetric NCA steps."""
    def rematerialized_update(current_state: jax.Array, step_key: jax.Array) -> jax.Array:
        return cell_update_3d(
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

    state_ncdhw, _ = jax.lax.scan(scan_step, state_ncdhw, step_keys)
    return state_ncdhw


def render_selected_plane_3d(
    state_ncdhw: jax.Array, axis: str = "z", index: int | None = None
) -> jax.Array:
    """Extract one spatial RGBA face from a cubic NCDHW state.

    ``axis`` is ``z`` (depth), ``y`` (height), or ``x`` (width). With a cubic
    volume every option yields a regular 32×32 RGBA training image.
    """
    if axis == "z":
        resolved_index = state_ncdhw.shape[2] // 2 if index is None else index
        return jnp.clip(state_ncdhw[:, :4, resolved_index, :, :], 0.0, 1.0)
    if axis == "y":
        resolved_index = state_ncdhw.shape[3] // 2 if index is None else index
        return jnp.clip(state_ncdhw[:, :4, :, resolved_index, :], 0.0, 1.0)
    if axis == "x":
        resolved_index = state_ncdhw.shape[4] // 2 if index is None else index
        return jnp.clip(state_ncdhw[:, :4, :, :, resolved_index], 0.0, 1.0)
    raise ValueError("axis must be one of 'x', 'y', or 'z'")


def seed_volume_3d(
    batch_size: int,
    state_channels: int,
    depth: int,
    height: int,
    width: int,
    seed_depth_index: int = 0,
    random_seed_density: float = 0.0,
    random_seed_key: jax.Array | None = None,
    seed_shape: str = "plane",
    seed_radius: float = 1.0,
) -> jax.Array:
    """Create a planar, spherical, or random-only living seed volume."""
    if state_channels < 4:
        raise ValueError("A visual 3D NCA requires RGB plus alpha")
    state = jnp.zeros((batch_size, state_channels, depth, height, width))
    if not 0 <= seed_depth_index < depth:
        raise ValueError("seed_depth_index must be inside the volume")
    y, x = height // 2, width // 2
    if seed_shape == "none":
        # Random seed cells below are the sole initial living cells.
        pass
    elif seed_shape == "plane":
        state = state.at[:, 3:, seed_depth_index, y - 1 : y + 2, x - 1 : x + 2].set(1.0)
    elif seed_shape == "sphere":
        z_grid, y_grid, x_grid = jnp.ogrid[:depth, :height, :width]
        sphere = (
            (z_grid - seed_depth_index) ** 2
            + (y_grid - y) ** 2
            + (x_grid - x) ** 2
            <= seed_radius**2
        )
        state = state.at[:, :3].set(jnp.where(sphere[None, None], 1.0, state[:, :3]))
        state = state.at[:, 3:].set(jnp.where(sphere[None, None], 1.0, state[:, 3:]))
    else:
        raise ValueError("seed_shape must be 'none', 'plane', or 'sphere'")
    if random_seed_density > 0.0:
        if random_seed_key is None:
            random_seed_key = jax.random.PRNGKey(0)
        random_mask = jax.random.uniform(
            random_seed_key, (batch_size, depth, height, width)
        ) < random_seed_density
        state = state.at[:, 3:].set(
            jnp.where(random_mask[:, None], 1.0, state[:, 3:])
        )
    return state
