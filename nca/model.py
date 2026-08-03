import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable


def sample_continuous_edges(
    state: jnp.ndarray, edge_pos: jnp.ndarray
) -> jnp.ndarray:
    """Bilinearly sample an NHWC state at normalized edge coordinates.

    Args:
        state: State with shape ``(B, H, W, C)``.
        edge_pos: Coordinates with shape ``(B, H, W, K, 2)`` in ``[-1, 1]``;
            the last dimension is ``(x, y)``.
    Returns:
        Samples with shape ``(B, H, W, K, C)``.
    """
    batch, height, width, _ = state.shape
    x = (edge_pos[..., 0] + 1.0) * (width - 1) / 2.0
    y = (edge_pos[..., 1] + 1.0) * (height - 1) / 2.0
    x = jnp.clip(x, 0.0, width - 1.0)
    y = jnp.clip(y, 0.0, height - 1.0)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, width - 1)
    y1 = jnp.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    batch_idx = jnp.arange(batch)[:, None, None, None]

    top_left = state[batch_idx, y0, x0]
    top_right = state[batch_idx, y0, x1]
    bottom_left = state[batch_idx, y1, x0]
    bottom_right = state[batch_idx, y1, x1]
    wx = wx[..., None]
    wy = wy[..., None]
    return (
        (1.0 - wx) * (1.0 - wy) * top_left
        + wx * (1.0 - wy) * top_right
        + (1.0 - wx) * wy * bottom_left
        + wx * wy * bottom_right
    )


class UpdateModel(nn.Module):
    model_output_len: int = 16
    perception_method: str = "sobel"
    nonlocal_connections: bool = False
    nonlocal_mode: str = "global"
    nonlocal_token_grid: int = 8
    nonlocal_attention_dim: int = 32
    edge_count: int = 4
    edge_state_dim: int = 16
    edge_step_size: float = 0.05
    edge_state_step_size: float = 0.05
    edge_message_scale: float = 0.25
    pokemon_vocab_size: int = 0
    pokemon_embedding_dim: int = 32
    kernel_init: Callable = nn.initializers.glorot_uniform

    def setup(self):
        """Initialize the model layers."""
        if self.perception_method == "learned":
            self.perception = nn.Conv(
                3 * self.model_output_len,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_init=self.kernel_init(),
            )
        if self.nonlocal_connections:
            if self.nonlocal_mode == "token_attention":
                self.query_projection = nn.Dense(self.nonlocal_attention_dim)
                self.key_projection = nn.Dense(self.nonlocal_attention_dim)
                self.value_projection = nn.Dense(self.nonlocal_attention_dim)
                self.context_gate = nn.Dense(
                    1,
                    bias_init=nn.initializers.constant(-4.0),
                )
            elif self.nonlocal_mode == "moving_edges":
                self.edge_net = nn.Sequential(
                    [
                        nn.Dense(64),
                        nn.relu,
                        nn.Dense(self.edge_state_dim + 2),
                    ]
                )
            else:
                self.global_projection = nn.Dense(32)
        if self.pokemon_vocab_size > 0:
            self.pokemon_embedding = nn.Embed(
                num_embeddings=self.pokemon_vocab_size,
                features=self.pokemon_embedding_dim,
            )
        self.conv_1 = nn.Conv(
            128,
            kernel_size=(1, 1),
            padding="SAME",
            kernel_init=self.kernel_init(),
        )
        self.conv_2 = nn.Conv(
            self.model_output_len,
            kernel_size=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )

    def __call__(
        self,
        perception_vector: jnp.ndarray,
        pokemon_ids: jnp.ndarray | None = None,
        state_grid: jnp.ndarray | None = None,
        owner_alive: jnp.ndarray | None = None,
        edge_pos: jnp.ndarray | None = None,
        edge_state: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply the model to an input tensor.

        Args:
            perception_vector: A 4D tensor representing the input data with shape (batch_size, channels, height, width).

        Returns:
            A 2D tensor representing the output data with shape (batch_size, output_len).
        """

        if self.perception_method == "learned":
            perception_vector = self.perception(perception_vector)

        next_edge_values = None
        if self.nonlocal_connections:
            if self.nonlocal_mode == "token_attention":
                batch, height, width, _ = perception_vector.shape
                tokens = jax.image.resize(
                    perception_vector,
                    (batch, self.nonlocal_token_grid, self.nonlocal_token_grid,
                     perception_vector.shape[-1]),
                    method="linear",
                )
                tokens = tokens.reshape(batch, -1, perception_vector.shape[-1])
                queries = self.query_projection(perception_vector)
                keys = self.key_projection(tokens)
                values = self.value_projection(tokens)
                scores = jnp.einsum("bhwd,bnd->bhwn", queries, keys)
                scores = scores / jnp.sqrt(float(self.nonlocal_attention_dim))
                weights = nn.softmax(scores, axis=-1)
                global_context = jnp.einsum("bhwn,bnd->bhwd", weights, values)
                gate = nn.sigmoid(self.context_gate(perception_vector))
                global_context = gate * global_context
            elif self.nonlocal_mode == "moving_edges":
                if state_grid is None or edge_pos is None or edge_state is None:
                    raise ValueError(
                        "moving_edges requires state_grid, edge_pos, and edge_state"
                    )
                if owner_alive is None:
                    owner_alive = jnp.ones(
                        perception_vector.shape[:3], dtype=perception_vector.dtype
                    )
                edge_owner = jnp.broadcast_to(
                    perception_vector[..., None, :],
                    perception_vector.shape[:3]
                    + (self.edge_count, perception_vector.shape[-1]),
                )
                edge_input = jnp.concatenate([edge_owner, edge_state], axis=-1)
                edge_delta = self.edge_net(edge_input)
                displacement = self.edge_step_size * jnp.tanh(
                    edge_delta[..., :2]
                )
                next_position = jnp.clip(
                    edge_pos + displacement, -1.0, 1.0
                )
                next_edge_state = edge_state + self.edge_state_step_size * jnp.tanh(
                    edge_delta[..., 2:]
                )
                next_edge_state = 4.0 * jnp.tanh(next_edge_state / 4.0)
                owner_gate = owner_alive[..., None, None]
                next_position = jnp.where(owner_gate, next_position, edge_pos)
                next_edge_state = jnp.where(owner_gate, next_edge_state, edge_state)
                sampled = sample_continuous_edges(
                    jnp.transpose(state_grid, (0, 2, 3, 1)), next_position
                )
                source_alive = jnp.clip(sampled[..., 3:4], 0.0, 1.0)
                weights = nn.softmax(next_edge_state[..., 0], axis=-1)
                received = jnp.sum(
                    sampled * source_alive * weights[..., None], axis=3
                )
                received = self.edge_message_scale * jnp.tanh(received)
                received = received * owner_alive[..., None]
                perception_vector = jnp.concatenate(
                    [perception_vector, received], axis=-1
                )
                next_edge_values = (
                    next_position,
                    next_edge_state,
                )
            else:
                # Every cell receives a learned summary of the entire grid.
                global_context = jnp.mean(
                    perception_vector, axis=(1, 2), keepdims=True
                )
                global_context = self.global_projection(global_context)
                global_context = jnp.broadcast_to(
                    global_context,
                    perception_vector.shape[:3] + (global_context.shape[-1],),
                )
            if self.nonlocal_mode != "moving_edges":
                perception_vector = jnp.concatenate(
                    [perception_vector, global_context], axis=-1
                )

        if self.pokemon_vocab_size > 0:
            if pokemon_ids is None:
                raise ValueError("pokemon_ids are required for conditional models")
            embedding = self.pokemon_embedding(pokemon_ids)
            embedding = embedding[:, None, None, :]
            embedding = jnp.broadcast_to(
                embedding,
                perception_vector.shape[:3] + (self.pokemon_embedding_dim,),
            )
            perception_vector = jnp.concatenate(
                [perception_vector, embedding], axis=-1
            )

        x = self.conv_1(perception_vector)
        x = nn.relu(x)
        ds = self.conv_2(x)
        if next_edge_values is not None:
            return ds, *next_edge_values
        return ds
