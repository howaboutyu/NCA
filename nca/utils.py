import io

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

from glob import glob
import tempfile
import cv2  # type: ignore
import os

try:
    from moviepy import ImageSequenceClip  # type: ignore
except ImportError:  # pragma: no cover - exercised with MoviePy 1.x
    from moviepy.editor import ImageSequenceClip  # type: ignore

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

    with open(filename, "wb") as output_file:
        output_file.write(encode_gif(images, fps))


def encode_gif(images: Union[List[Any], np.ndarray], fps: int = 10) -> bytes:
    """Encode RGB frames as one looping animated GIF without console output."""
    frames = []
    for image in images:
        frame = np.asarray(image)
        if frame.size and frame.max() <= 1.0:
            frame = frame * 255.0
        frames.append(np.clip(frame, 0.0, 255.0).astype(np.uint8))
    if not frames:
        raise ValueError("Cannot encode an empty GIF")
    buffer = io.BytesIO()
    imageio.mimsave(buffer, frames, format="GIF", duration=1.0 / fps, loop=0)
    return buffer.getvalue()


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


def download_pokemon_emoji(
    output_dir: str = "data/pokemon_emoji",
) -> dict:
    git_url = "https://github.com/Templarian/slack-emoji-pokemon"

    # download pokemon emoji
    os.system(f"git clone {git_url} {output_dir}")

    # emoji dir
    emoji_dir = f"{output_dir}/emojis"

    emoji_list = glob(f"{emoji_dir}/*.png")

    emoji_dict = {}

    for emoji in emoji_list:
        emoji_name = emoji.split("/")[-1].split(".")[0]
        emoji_dict[emoji_name] = emoji

    return emoji_dict
