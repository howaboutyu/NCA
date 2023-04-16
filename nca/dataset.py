from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Callable
import numpy as np
import jax


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
