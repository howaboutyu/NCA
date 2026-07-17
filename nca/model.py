import jax.numpy as jnp
import flax.linen as nn
from typing import Callable


class UpdateModel(nn.Module):
    model_output_len: int = 16
    perception_method: str = "sobel"
    nonlocal_connections: bool = False
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
            self.global_projection = nn.Dense(32)
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

        if self.perception_method == "learned":
            perception_vector = self.perception(perception_vector)

        if self.nonlocal_connections:
            # Every cell receives a learned summary of the entire grid. This
            # is a low-cost long-range connection, rather than a wider local
            # convolution: the summary is pooled globally and broadcast back.
            global_context = jnp.mean(perception_vector, axis=(1, 2), keepdims=True)
            global_context = self.global_projection(global_context)
            global_context = jnp.broadcast_to(
                global_context,
                perception_vector.shape[:3] + (global_context.shape[-1],),
            )
            perception_vector = jnp.concatenate(
                [perception_vector, global_context], axis=-1
            )

        x = self.conv_1(perception_vector)
        x = nn.relu(x)
        ds = self.conv_2(x)
        return ds
