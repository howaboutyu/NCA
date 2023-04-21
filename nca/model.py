import jax.numpy as jnp
import flax.linen as nn


class UpdateModel(nn.Module):
    model_output_len: int = 16

    def setup(self):
        """Initialize the model layers."""
        self.dense1 = nn.Dense(128)
        # self.dense2 = nn.Dense(128)

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
        x = self.dense1(perception_vector)
        x = nn.relu(x)
        # x = self.dense2(x)
        # x = nn.relu(x)
        ds = self.dense_final(x)
        return ds
