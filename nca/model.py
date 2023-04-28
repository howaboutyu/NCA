import jax.numpy as jnp
import flax.linen as nn
from typing import Callable


class UpdateModel(nn.Module):
    model_output_len: int = 16
    kernel_init: Callable = nn.initializers.glorot_uniform

    def setup(self):
        """Initialize the model layers."""
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

    def __call__(self, perception_vector: jnp.ndarray) -> jnp.ndarray:
        """Apply the model to an input tensor.

        Args:
            perception_vector: A 4D tensor representing the input data with shape (batch_size, channels, height, width).

        Returns:
            A 2D tensor representing the output data with shape (batch_size, output_len).
        """

        x = self.conv_1(perception_vector)
        x = nn.relu(x)
        ds = self.conv_2(x)
        return ds
