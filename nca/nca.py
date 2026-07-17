import jax
import jax.numpy as jnp

import numpy as np
from typing import Any, Callable, Optional, Tuple, Union


def create_perception_kernel(
    input_size: int = 16, output_size: int = 16, use_oihw_layout: bool = True
) -> tuple:
    """Create a perception kernel for edge detection.

    Args:
        input_size: The size of the input image along the third dimension.
            Default is 48.
        output_size: The number of output channels in the Sobel operator.
            Default is 16.
        use_oihw_layout: Whether to use the oihw layout for the output kernels.
            Default is True.

    Returns:
        A tuple containing two 4D JAX arrays representing the Sobel operator
        for edge detection in the x and y directions.
    """
    # Initialize kernel using NHWC layout
    kernel = jnp.zeros((3, 3, input_size, output_size), dtype=jnp.float32)

    # Compute Sobel operator for x direction
    kernel_x = (
        kernel
        + jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[
            :, :, jnp.newaxis, jnp.newaxis
        ]
    )

    # Compute Sobel operator for y direction
    kernel_y = jnp.transpose(kernel_x, (1, 0, 2, 3))

    if use_oihw_layout:
        # Transpose kernels to IOHW layout
        kernel_x = jnp.transpose(kernel_x, (3, 2, 0, 1))
        kernel_y = jnp.transpose(kernel_y, (3, 2, 0, 1))

    return kernel_x / 8.0, kernel_y / 8.0


def create_second_derivative_kernels(
    input_size: int = 16, output_size: int = 16, use_oihw_layout: bool = True
) -> tuple:
    """Create finite-difference kernels for the pure second derivatives."""
    kernel_xx = jnp.zeros((3, 3, input_size, output_size), dtype=jnp.float32)
    kernel_xx = kernel_xx + jnp.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])[
        :, :, jnp.newaxis, jnp.newaxis
    ]
    kernel_yy = kernel_xx.transpose(1, 0, 2, 3)

    if use_oihw_layout:
        kernel_xx = jnp.transpose(kernel_xx, (3, 2, 0, 1))
        kernel_yy = jnp.transpose(kernel_yy, (3, 2, 0, 1))

    return kernel_xx, kernel_yy


def perceive(
    state_grid: jnp.ndarray,
    kernel_x: jnp.ndarray,
    kernel_y: jnp.ndarray,
    method: str = "sobel",
    kernel_xx: Optional[jnp.ndarray] = None,
    kernel_yy: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Perceive an input state grid using edge detection.

    Args:
        state_grid: A 4D JAX array representing the input state grid.
            Must be in NCHW layout.
        kernel_x: A 4D JAX array representing the Sobel operator for
            edge detection in the x direction. Must be in OIHW layout.
        kernel_y: A 4D JAX array representing the Sobel operator for
            edge detection in the y direction. Must be in OIHW layout.

    Returns:
        A 4D JAX array representing the perceived state grid.
        Has the same shape as the input state grid, with three times
        as many channels. The first set of channels correspond to the
        input state grid, and the next two sets correspond to the
        gradients in the x and y directions, respectively.
    """
    if method == "sobel":
        # Original implementation: two separate convolutions.
        grad_x = jax.lax.conv(state_grid, kernel_x, (1, 1), "SAME")
        grad_y = jax.lax.conv(state_grid, kernel_y, (1, 1), "SAME")
    elif method == "sobel_fused":
        # Concatenating output channels lets XLA use one convolution while
        # preserving the exact feature ordering of the original path.
        fused_kernel = jnp.concatenate([kernel_x, kernel_y], axis=0)
        gradients = jax.lax.conv(state_grid, fused_kernel, (1, 1), "SAME")
        channels = state_grid.shape[1]
        grad_x, grad_y = gradients[:, :channels], gradients[:, channels:]
    elif method == "sobel_second":
        if kernel_xx is None or kernel_yy is None:
            raise ValueError("sobel_second requires second-derivative kernels")
        fused_kernel = jnp.concatenate(
            [kernel_x, kernel_y, kernel_xx, kernel_yy], axis=0
        )
        derivatives = jax.lax.conv(state_grid, fused_kernel, (1, 1), "SAME")
        channels = state_grid.shape[1]
        grad_x = derivatives[:, :channels]
        grad_y = derivatives[:, channels : 2 * channels]
        grad_xx = derivatives[:, 2 * channels : 3 * channels]
        grad_yy = derivatives[:, 3 * channels :]
        return jnp.concatenate(
            [state_grid, grad_x, grad_y, grad_xx, grad_yy], axis=1
        )
    else:
        raise ValueError(f"Unsupported fixed perception method: {method}")

    # Concatenate the state grid with the gradients along the channel axis
    perceived_grid = jnp.concatenate([state_grid, grad_x, grad_y], axis=1)

    return perceived_grid  # -> NCHW


def cell_update(
    key: jax.Array,
    state_grid: jax.Array,
    model_fn: Any,
    params: jax.Array,
    kernel_x: jax.Array,
    kernel_y: jax.Array,
    update_prob: float = 0.5,
    perception_method: str = "sobel",
    kernel_xx: Optional[jax.Array] = None,
    kernel_yy: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """
    Cell update function to perform the update on the given state grid.

    Args:
        key: A JAX array representing the random key.
        state_grid: A JAX array representing the input state grid.
        model_fn: The model function to be applied to the perceived grid.
        params: A JAX array representing the model parameters.
        kernel_x: A JAX array representing the Sobel operator for
            edge detection in the x direction.
        kernel_y: A JAX array representing the Sobel operator for
            edge detection in the y direction.
        update_prob: The probability of cell update. Default is 0.5.

    Returns:
        A JAX array representing the updated state grid.
    """
    pre_alive_mask = alive_masking(state_grid[:, 3, :, :])

    # Fixed perception operates on NCHW. Learned perception is a trainable
    # model layer and receives the raw state in NHWC.
    if perception_method == "learned":
        model_input = jnp.transpose(state_grid, (0, 2, 3, 1))
    else:
        perceived_grid = perceive(
            state_grid,
            kernel_x,
            kernel_y,
            method=perception_method,
            kernel_xx=kernel_xx,
            kernel_yy=kernel_yy,
        )
        model_input = jnp.transpose(perceived_grid, (0, 2, 3, 1))

    ds = model_fn.apply(params, model_input)

    # Stochastic update
    rand_mask = jax.random.uniform(key, shape=ds.shape[:-1]) < update_prob
    rand_mask = rand_mask.astype(jnp.float32)

    ds = ds * rand_mask[..., jnp.newaxis]

    # Transpose: NHWC -> NCHW
    ds = jnp.transpose(ds, (0, 3, 1, 2))

    state_grid = state_grid + ds

    post_alive_mask = alive_masking(state_grid[:, 3, :, :])
    alive_mask = pre_alive_mask * post_alive_mask
    alive_mask = jnp.expand_dims(alive_mask, 1)

    state_grid = alive_mask.astype(jnp.float32) * state_grid

    return state_grid


def alive_masking(
    alive_state: jnp.ndarray,
    alive_threshold: float = 0.1,
    window_shape: tuple = (1, 3, 3),
    window_stride: tuple = (1, 1, 1),
) -> jnp.ndarray:
    """Applies alive masking to the input state array to identify "dead" cells.

    Args:
        alive_state: The input state array of shape (batch_size, num_channels, height, width).
        alive_threshold: The threshold value to use for the alive mask.
        window_shape: The shape of the window to use for max pooling.
        window_stride: The stride of the window to use for max pooling.

    Returns:
        An array of shape (batch_size, num_channels, height, width) where each element is either 1.0 (alive) or 0.0 (dead).
    """
    # Max pooling
    max_pool = jax.lax.reduce_window(
        alive_state,
        -jnp.inf,
        jax.lax.max,
        window_strides=window_stride,
        window_dimensions=window_shape,
        padding="SAME",
    )

    # If there are not "mature" cells in the 3x3 window, then the cell is "dead"
    alive_mask = max_pool > alive_threshold

    alive_mask = alive_mask.astype(jnp.float32)

    return alive_mask
