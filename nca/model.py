import jax.numpy as jnp
import flax.linen as nn
from typing import List


class UpdateModel(nn.Module):
    model_output_len: int = 16
    num_layers: int = 3  # number of dense->relu layers

    def setup(self):
        """Initialize the model layers."""

        self.dense_array = [
            nn.Dense(64, kernel_init=nn.initializers.lecun_normal())
            for _ in range(self.num_layers)
        ]

        self.dense_final = nn.Dense(
            self.model_output_len, kernel_init=nn.initializers.zeros
        )

    def __call__(self, perception_vector: jnp.ndarray) -> jnp.ndarray:
        """Apply the model to an input tensor.

        Args:
            perception_vector: A 4D tensor representing the input data with shape (batch_size, channels, height, width).

        Returns:
            A 2D tensor representing the output data with shape (batch_size, output_len).
        """
        x = perception_vector

        for dense in self.dense_array:
            x = dense(x)
            x = nn.relu(x)
        ds = self.dense_final(x)
        return ds
