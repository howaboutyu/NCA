import jax.numpy as jnp


import jax.numpy as jnp


def NCWH_to_NHWC(x: jnp.ndarray) -> jnp.ndarray:
    """Converts an array from the NCWH format to the NHWC format.

    Args:
        x: The input array in NCWH format.

    Returns:
        The output array in NHWC format.
    """
    return jnp.transpose(x, (0, 2, 3, 1))


def NHWC_to_NCWH(x: jnp.ndarray) -> jnp.ndarray:
    """Converts an array from the NHWC format to the NCWH format.

    Args:
        x: The input array in NHWC format.

    Returns:
        The output array in NCWH format.
    """
    return jnp.transpose(x, (0, 3, 1, 2))


def alpha_mask(x: jnp.ndarray) -> jnp.ndarray:
    """Masks an array using the alpha channel.

    Args:
        x: The input array in NHWC format.

    Returns:
        The output array in NHWC format.
    """
    # clip to [0, 1]
    x = jnp.clip(x, 0, 1)
    return x[:, :, :, :3] * x[:, :, :, 3:4]
