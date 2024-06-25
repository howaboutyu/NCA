from dataclasses import dataclass, field
from typing import Tuple, Any, Dict
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
        self.seed_state = self.get_seed_state()

        # old:  set chemical channels 1, at the center of the grid
        # self.seed_state[3:, self.dimensions[0] // 2, self.dimensions[1] // 2] = 1.0

        self.pool = np.asarray([self.seed_state] * self.pool_size)

    def get_seed_state(self):
        seed_state = (
            np.random.randint(
                0,
                2,
                size=(self.model_output_len, self.dimensions[0], self.dimensions[1]),
            )
            * 1.0
        )

        seed_state[:3] = 0.0

        return seed_state

    def sample(
        self, key: Any, damage: bool = False, K: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        # sample a batch of random indices from the pool
        indices = jax.random.randint(
            key, shape=(self.batch_size,), minval=0, maxval=self.pool_size
        )
        indices_np = np.asarray(indices)

        updated_pool = self.pool[indices_np]

        return updated_pool, indices_np

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
        # target[:, :, 0:5, 0:5] = 1.0

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


@dataclass
class XorDataGenerator:
    pool_size: int
    batch_size: int
    dimensions: Tuple[Any, ...]
    model_output_len: int
    seed_state: np.ndarray = field(init=False)
    pool: np.ndarray = field(init=False)
    pool_targets: np.ndarray = field(init=False)
    xor_dict: Dict[Tuple[int, int], int] = field(init=False)

    def __post_init__(self):
        # self.pool = np.asarray([self.seed_state] * self.pool_size)

        # self.pool_condition = np.random.rand(0, num_classes + 1, size=(self.pool_size, ))

        self.xor_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
        # seed_states = self.get_seed_state()
        seed_states = []
        xor_list = list(self.xor_dict)
        self.xor_list= xor_list
        pool_targets = []
        for pool_id in range(self.pool_size):
            rand_class = np.random.randint(0, 4)

            input = xor_list[rand_class]
            output = self.xor_dict[input]
            

            rand_state = (
                np.random.randint(
                    0,
                    2,
                    size=(
                        self.model_output_len,
                        self.dimensions[0],
                        self.dimensions[1],
                    ),
                )
                * 1.0 
            )

            rand_target_state = (
                np.zeros(
                    (
                        self.model_output_len,
                        self.dimensions[0],
                        self.dimensions[1],
                    ),
                )
                * 1.0
            )

            if input[0] == 1:
                rand_state[:, 0:5, 0:5] = 1.0  # top left
            else:
                rand_state[:, 0:5, 0:5] = 0.5  # top left
                rand_state[4:, 0:5, 0:5] = 1.0  # top left

            if input[1] == 1:
                rand_state[:, -5:, 0:5] = 1.0  # top right
            else:
                rand_state[:, -5:, 0:5] = 0.5  # top right
                rand_state[4:, -5:, 0:5] = 1.0  # top right

            if output == 1:
                rand_target_state[0:4, -5:, -5:] = 1.0
            else:
                rand_target_state[0:4, 0:5,-5:] = 1.0

            seed_states.append(rand_state)
            pool_targets.append(rand_target_state)

        self.seed_states = np.asarray(seed_states)
        self.pool = self.seed_states.copy()
        self.pool_targets = np.asarray(pool_targets)
        

        #import pdb; pdb.set_trace()


    def get_seed_state(self):
        seed_state = (
            np.random.randint(
                0,
                2,
                size=(self.model_output_len, self.dimensions[0], self.dimensions[1]),
            )
            *1.0 
        )



        rand_id = np.random.randint(0, 4)

        input = self.xor_list[rand_id]

        if input[0] == 1:
            seed_state[:, 0:5, 0:5] = 1.0
        else:
            seed_state[:, 0:5, 0:5] = 0.5
            seed_state[4:, 0:5, 0:5] = 1.0

        if input[1] == 1:
            seed_state[:, -5:, 0:5] = 1.0
        else:
            seed_state[:, -5:, 0:5] = 0.5
            seed_state[4:, -5:, 0:5] = 1.0

        target = self.xor_dict[input]

        target_state= np.zeros((self.model_output_len, self.dimensions[0], self.dimensions[1]))

        if target == 1:
            target_state[0:4, -5:, -5:] = 1.0
        elif target == 0:
            target_state[0:4, 0:5, -5:] = 1.0



        return seed_state, target_state

    def sample(
        self, key: Any, damage: bool = False, K: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        # sample a batch of random indices from the pool
        indices = jax.random.randint(
            key, shape=(self.batch_size,), minval=0, maxval=self.pool_size
        )
        indices_np = np.asarray(indices)
        updated_pool = np.asarray([self.pool[p_id] for   p_id in indices_np])




        return updated_pool, indices_np

    def update_pool(self, indices: Any, new_states: np.ndarray):
        #import pdb; pdb.set_trace()
        self.pool[indices] = new_states

    def get_target(self, state_grid_indices) -> jax.Array:
        # Load the image with alpha channel

        all_targets = []
        for indices in state_grid_indices:
            target_map = self.pool_targets[indices]
            
            all_targets.append(target_map)

        target = np.asarray(all_targets)
        #import pdb; pdb.set_trace()
        #target = np.transpose(target, (0, 3, 1, 2))
        # target[:, :, 0:5, 0:5] = 1.0
        return target 
        # return jnp.asarray(target)

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


if __name__ == "__main__":
    xor = XorDataGenerator(
        pool_size=8, batch_size=4, dimensions=(56, 56), model_output_len=16
    )

    xor.get_target(
        [
            0,
            1,
            2,
        ]
    )
