import jax
import jax.numpy as jnp

# REMOVE
import numpy as np

import matplotlib.pyplot as plt
import cv2


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

    return kernel_x, kernel_y


def perceive(
    state_grid: jnp.ndarray, kernel_x: jnp.ndarray, kernel_y: jnp.ndarray
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
    # Compute gradients using convolution with Sobel operators
    grad_x = jax.lax.conv(state_grid, kernel_x, (1, 1), "SAME")  # -> NCHW
    grad_y = jax.lax.conv(state_grid, kernel_y, (1, 1), "SAME")  # -> NCHW

    # Concatenate the state grid with the gradients along the channel axis
    perceived_grid = jnp.concatenate([state_grid, grad_x, grad_y], axis=1)

    return perceived_grid  # -> NCHW


def cell_update(key, state_grid, model_fn, params, kernel_x, kernel_y, update_prob=0.5):
    perceived_grid = perceive(state_grid, kernel_x, kernel_y)

    # NCHW -> NHWC
    perceived_grid = jnp.transpose(perceived_grid, (0, 2, 3, 1))
    ds = model_fn(params, perceived_grid)  # -> NHWC

    # Stochastic update
    rand_mask = jax.random.uniform(key, shape=ds.shape[:-1]) < update_prob
    ds = ds * rand_mask[..., jnp.newaxis]  # -> NHWC

    # HWCN -> NCHW
    ds = jnp.transpose(ds, (0, 3, 1, 2))

    state_grid = state_grid + ds

    # Alive masking
    state_grid = alive_masking(state_grid)

    return state_grid


def alive_masking(state_grid, alive_threshold=0.1):
    # This function is NCWH
    # Alive masking
    win_size = (1, 3, 3)
    win_stride = (1, 1, 1)

    # alive state is the 4th channel
    alive_state = state_grid[:, 3, :, :]

    # Max pooling
    alive = jax.lax.reduce_window(
        alive_state,
        -jnp.inf,
        jax.lax.max,
        window_strides=win_stride,
        window_dimensions=win_size,
        padding="SAME",
    )

    # if there are not "mature" cells in the 3x3 window, then the cell is "dead"
    alive = alive > alive_threshold

    alive = jnp.expand_dims(alive, axis=1)

    state_grid = state_grid * alive
    return state_grid


"""
kernel_x, kernel_y = create_perception_kernel(use_iohw_layout=True)

# state_grid = jnp.zeros((1, 200, 198, 3), dtype=jnp.float32)
state_grid = cv2.imread("emoji_imgs/skier.png") / 255.0
state_grid = jnp.expand_dims(state_grid, axis=0)
state_grid = jnp.transpose(state_grid, (0, 3, 1, 2))  # NHWC -> NCHW

perceived_grid = perceive(state_grid, kernel_x, kernel_y)


plt.figure(figsize=(10, 10))
plt.imshow(np.array(perceived_grid)[0, 4, :, :])
plt.show()
"""
