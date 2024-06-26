from dataclasses import dataclass, field
from typing import Tuple, Any
import numpy as np
import jax
import jax.numpy as jnp
import cv2  # type: ignore
import tensorflow as tf  # type: ignore
import keras_cv

from nca.utils import NCHW_to_NHWC, NHWC_to_NCHW

Array = Any


@dataclass
class NCADataGenerator:
    pool_size: int
    batch_size: int
    dimensions: Tuple[Any, ...]
    model_output_len: int
    seed_state: np.ndarray = field(init=False)
    pool: np.ndarray = field(init=False)

    def __post_init__(self):
        self.seed_state = np.zeros(
            (self.model_output_len, self.dimensions[0], self.dimensions[1])
        )

        # set chemical channels 1, at the center of the grid
        self.seed_state[3:, self.dimensions[0] // 2, self.dimensions[1] // 2] = 1.0

        self.pool = np.asarray([self.seed_state] * self.pool_size)

    def sample(
        self, key: Any, damage: bool = False, K: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        # sample a batch of random indices from the pool
        indices = jax.random.randint(
            key, shape=(self.batch_size,), minval=0, maxval=self.pool_size
        )
        indices_np = np.asarray(indices)

        return self.pool[indices_np], indices_np

    def update_pool(self, indices: Any, new_states: np.ndarray):
        self.pool[indices] = new_states

    def get_target(self, filename: str) -> jax.Array:
        # Load the image with alpha channel
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # Check if the image has 4 channels
        if img is None or img.shape[2] != 4:
            raise ValueError("Image must have 4 channels")

        # resize first 2xself.dimensions
        # img = cv2.resize(img, (2 * self.dimensions[0], 2 * self.dimensions[1]), interpolation=cv2.INTER_NEAREST)

        # Pad the image
        pad_width = ((25, 25), (25, 25), (0, 0))
        img = np.pad(img, pad_width, mode="constant", constant_values=0)

        # Resize the image to the target dimensions
        img = cv2.resize(img, self.dimensions, interpolation=cv2.INTER_NEAREST)

        # Convert image to float32
        img = img.astype(np.float32)

        # Apply alpha channel to color channels and normalize the color channels
        alpha = img[..., -1] > 1.0
        alpha = alpha.astype(np.float32)
        img[..., :3] = img[..., :3] * alpha[..., np.newaxis] / 255.0
        img[..., -1] = alpha
        img[..., [0, 1, 2]] = img[..., [2, 1, 0]]

        # Duplicate the image for batch size and transpose the axes
        target = np.asarray([img] * self.batch_size)
        target = np.transpose(target, (0, 3, 1, 2))

        return jnp.asarray(target)

    @staticmethod
    def random_cutout_rect(
        img_nchw: Array,
        height_factor: float = 0.1,
        width_factor: float = 0.1,
        seed: int = 10,
    ):
        img_nhwc = NCHW_to_NHWC(img_nchw)
        img_nhwc = keras_cv.layers.RandomCutout(height_factor, width_factor, seed=seed)(
            img_nhwc
        )

        img_nchw = NHWC_to_NCHW(img_nhwc)

        return img_nchw

    @staticmethod
    def random_cutout_circle(img_nchw: Array, seed: int):
        img_nhwc = NCHW_to_NHWC(img_nchw)

        n, h, w, _ = img_nhwc.shape

        x = tf.linspace(-1.0, 1.0, w)[None, None, :]
        y = tf.linspace(-1.0, 1.0, h)[None, :, None]
        center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5, seed=seed)
        r = tf.random.uniform([n, 1, 1], 0.05, 0.2, seed=seed)
        x, y = (x - center[0]) / r, (y - center[1]) / r
        mask: tf.Tensor = tf.cast(x * x + y * y < 1.0, tf.float32)
        img_masked = img_nhwc * (1.0 - mask[..., tf.newaxis])
        img_masked = np.asarray(img_masked)
        img_masked_nchw = NHWC_to_NCHW(img_masked)
        return img_masked_nchw
