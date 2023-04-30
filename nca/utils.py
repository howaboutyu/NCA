import jax.numpy as jnp
import numpy as np
from moviepy.editor import ImageSequenceClip  # type: ignore
import tempfile
from glob import glob
import cv2  # type: ignore

from typing import List, Any, Union


Array = Union[np.ndarray, jnp.ndarray]


def NCHW_to_NHWC(x: Array) -> Array:
    """Converts an array from the NCWH format to the NHWC format.

    Args:
        x: The input array in NCWH format.

    Returns:
        The output array in NHWC format.
    """
    if isinstance(x, np.ndarray):
        return np.transpose(x, (0, 2, 3, 1))
    else:
        return jnp.transpose(x, (0, 2, 3, 1))


def NHWC_to_NCHW(x: Array) -> Array:
    """Converts an array from the NHWC format to the NCHW format.

    Args:
        x: The input array in NHWC format.

    Returns:
        The output array in NCWH format.
    """
    if isinstance(x, np.ndarray):
        return np.transpose(x, (0, 3, 1, 2))
    else:
        return jnp.transpose(x, (0, 3, 1, 2))


def alpha_mask(x: Array) -> Array:
    """Masks an array using the alpha channel.

    Args:
        x: The input array in NHWC format.

    Returns:
        The output array in NHWC format.
    """
    # clip to [0, 1]
    x = jnp.clip(x, 0, 1)
    return x[:, :, :, :3] * x[:, :, :, 3:4]


# def state_grid_to_rgb(state_grid: jnp.ndarray) -> jnp.ndarray:
#    # extract the predicted RGB values and alpha channel from the state grid
#    state_grid = jnp.clip(state_grid, 0.0, 1.0)
#    alpha = state_grid[:, 3:4]
#    rgb = state_grid[:, :3]
#    rgb = rgb * alpha
#    return rgb


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
        images: A list of images with values in the range [0, 1]
        filename: The name of the GIF file.
        fps: The number of frames per second. Default is 10.

    """

    with tempfile.TemporaryDirectory() as tempdir:
        # write images to tempdir
        for i, image in enumerate(images):
            image = image * 255.0
            image = np.asarray(image[..., :3]).astype(np.uint8)
            image = np.squeeze(image)

            if image.shape[-1] != 1:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"{tempdir}/{str(i).zfill(5)}.png", image)

        image_files = glob(f"{tempdir}/*.png")
        image_files = sorted(image_files)  # , key=lambda x: int(x.split("/")[-1][:-4]))
        clip = ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(filename, fps=fps)


def mse(
    pred_rgb: Union[jnp.ndarray, np.ndarray],
    target: Union[jnp.ndarray, np.ndarray],
    reduce_mean: bool = True,
) -> jnp.ndarray:
    if reduce_mean == True:
        loss_value = jnp.mean(jnp.square(pred_rgb - target))
    else:
        loss_value = jnp.square(pred_rgb - target)

    return loss_value
