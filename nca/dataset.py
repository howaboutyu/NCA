from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Callable
import numpy as np
import jax
import cv2


@dataclass
class NCADataGenerator:
    pool_size: int
    batch_size: int
    dimensions: Tuple[int, int]
    model_output_len: int
    seed_state: np.ndarray = None
    pool: np.ndarray = None

    def __post_init__(self):
        self.seed_state = np.zeros(
            (self.dimensions[0], self.dimensions[1], self.model_output_len)
        )

        # set chemical channels 1, at the center of the grid
        self.seed_state[self.dimensions[0] // 2, self.dimensions[1] // 2, 3:] = 1.0

        self.pool = np.asarray([self.seed_state] * self.pool_size)

    def sample(self, key: np.ndarray) -> np.ndarray:
        # sample a batch of random indices from the pool
        indices = jax.random.randint(
            key, shape=(self.batch_size,), minval=0, maxval=self.pool_size
        )

        return self.pool[indices], indices

    def update_pool(self, indices: np.ndarray, new_states: np.ndarray):
        self.pool[indices] = new_states

    def get_target(self, filename: str) -> np.ndarray:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert img.shape[2] == 4, "Image must have 4 channels"
        alpha = img[..., -1] > 0
        img = img[..., :3] * alpha[..., np.newaxis]
        img = cv2.resize(img, self.dimensions)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        target = np.asarray([img] * self.batch_size)

        return target
