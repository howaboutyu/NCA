import jax.numpy as jnp
import numpy as np
from moviepy.editor import ImageSequenceClip  # type: ignore
import tempfile
from glob import glob
import cv2  # type: ignore

from typing import List, Any, Union


def NCHW_to_NHWC(x: jnp.ndarray) -> jnp.ndarray:
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


def state_grid_to_rgb(state_grid: jnp.ndarray) -> jnp.ndarray:
    # extract the predicted RGB values and alpha channel from the state grid
    alpha = state_grid[:, 3:4]
    alpha = jnp.clip(alpha, 0.0, 1.0)
    rgb = state_grid[:, :3]
    rgb = 1 -  alpha + rgb
    return rgb


def make_gif(
    images: Union[List[Any], np.ndarray], filename: str, fps: int = 10
) -> None:
    """Creates a movie from a list of images.

    Args:
        images: A list of images.
        filename: The name of the GIF file.
        fps: The number of frames per second. Default is 10.

    """

    with tempfile.TemporaryDirectory() as tempdir:
        # write images to tempdir
        for i, image in enumerate(images):
            image = np.asarray(image).astype(np.uint8)

            cv2.imwrite(f"{tempdir}/{i}.png", image)

        # create gif
        clip = ImageSequenceClip(glob(f"{tempdir}/*.png"), fps=fps)
        clip.write_gif(filename, fps=fps)


def make_video(
    images: Union[List[Any], np.ndarray], filename: str, fps: int = 10
) -> None:
    """Creates a movie from a list of images.

    Args:
        images: A list of images.
        filename: The name of the GIF file.
        fps: The number of frames per second. Default is 10.

    """

    with tempfile.TemporaryDirectory() as tempdir:
        # write images to tempdir
        for i, image in enumerate(images):
            image = np.asarray(image[..., :3]).astype(np.uint8)
            image = np.squeeze(image)

            cv2.imwrite(f"{tempdir}/{i}.png", image)

        clip = ImageSequenceClip(glob(f"{tempdir}/*.png"), fps=fps)
        clip.write_videofile(filename, fps=fps)
