import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable


class UpdateModel(nn.Module):
    model_output_len: int = 16
    perception_method: str = "sobel"
    nonlocal_connections: bool = False
    nonlocal_mode: str = "global"
    nonlocal_token_grid: int = 8
    nonlocal_attention_dim: int = 32
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
    ) -> jnp.ndarray:
        """Apply the model to an input tensor.

        Args:
            perception_vector: A 4D tensor representing the input data with shape (batch_size, channels, height, width).

        Returns:
            A 2D tensor representing the output data with shape (batch_size, output_len).
        """

        if self.perception_method == "learned":
            perception_vector = self.perception(perception_vector)

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
        return ds
