from dataclasses import dataclass, field
from typing import Tuple, Any
import numpy as np
import jax
import jax.numpy as jnp
import cv2  # type: ignore
import tensorflow as tf  # type: ignore

from nca.utils import NCHW_to_NHWC, NHWC_to_NCHW

Array = Any


@dataclass
class NCADataGenerator:
    pool_size: int
    batch_size: int
    dimensions: Tuple[Any, ...]
    model_output_len: int
    seed_density: float = 0.0
    seed_noise_density: float = 0.0
    seed_random_seed: int = 0
    seed_pattern: str = "single"
    seed_size: int = 11
    pokemon_targets: tuple = ()
    seed_state: np.ndarray = field(init=False)
    pool: np.ndarray = field(init=False)
    pool_pokemon_ids: np.ndarray = field(init=False)

    def __post_init__(self):
        if not 0.0 <= self.seed_density <= 1.0:
            raise ValueError("seed_density must be between 0.0 and 1.0")
        if not 0.0 <= self.seed_noise_density <= 1.0:
            raise ValueError("seed_noise_density must be between 0.0 and 1.0")
        if self.seed_pattern not in {"single", "random", "pokeball"}:
            raise ValueError("seed_pattern must be 'single', 'random', or 'pokeball'")
        if self.seed_size < 3 or self.seed_size % 2 == 0:
            raise ValueError("seed_size must be an odd integer >= 3")
        self.seed_state = np.zeros(
            (self.model_output_len, self.dimensions[0], self.dimensions[1])
        )

        if self.seed_pattern == "pokeball":
            self._initialize_pokeball()
        elif self.seed_density == 0.0 and self.seed_pattern == "single":
            # Original single-cell center seed.
            self.seed_state[
                3:, self.dimensions[0] // 2, self.dimensions[1] // 2
            ] = 1.0
        else:
            rng = np.random.default_rng(self.seed_random_seed)
            alive = rng.random(self.dimensions) < self.seed_density
            # Keep the seed usable even for very small grids/densities.
            if not np.any(alive):
                alive[self.dimensions[0] // 2, self.dimensions[1] // 2] = True
            self.seed_state[3:, alive] = 1.0

        if self.seed_noise_density > 0.0:
            rng = np.random.default_rng(self.seed_random_seed)
            noise_mask = rng.random(self.dimensions) < self.seed_noise_density
            # Preserve the Poké Ball itself; add noise around it.
            noise_mask &= self.seed_state[3] <= 0.0
            self.seed_state[:3, noise_mask] = rng.random(
                (3, int(np.count_nonzero(noise_mask)))
            )
            self.seed_state[3:, noise_mask] = 1.0

        self.pool = np.asarray([self.seed_state] * self.pool_size)
        self.pokemon_targets = tuple(self.pokemon_targets)
        if not self.pokemon_targets:
            self.pokemon_targets = (None,)
        self.pool_pokemon_ids = np.arange(self.pool_size, dtype=np.int32) % len(
            self.pokemon_targets
        )

    @property
    def pokemon_vocab_size(self) -> int:
        return len(self.pokemon_targets)

    def get_targets(self, fallback_filename: str) -> jax.Array:
        filenames = [filename or fallback_filename for filename in self.pokemon_targets]
        return jnp.asarray(
            np.stack([np.asarray(self.get_target(filename)) for filename in filenames])
        )

    def _initialize_pokeball(self):
        """Place a compact Poké Ball icon at the center of the seed grid."""
        h, w = self.dimensions
        radius = self.seed_size / 2.0
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = distance <= radius
        band = np.abs(yy - cy) <= max(1, self.seed_size // 10)
        button = distance <= max(1.0, self.seed_size / 7.0)

        # Channels 0:3 are the visible RGB seed; channel 3 is its alpha/living
        # channel. Hidden channels are initialized as living wherever the icon
        # is present, matching the existing NCA seed convention.
        self.seed_state[0, mask & (yy <= cy)] = 0.9  # red
        self.seed_state[1, mask & (yy <= cy)] = 0.05
        self.seed_state[2, mask & (yy <= cy)] = 0.05
        self.seed_state[0:3, mask & (yy > cy)] = 1.0  # white
        self.seed_state[0:3, mask & band] = 0.02  # black band
        self.seed_state[0:3, mask & button] = 1.0  # white center button
        self.seed_state[3:, mask] = 1.0

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
        # Keep augmentation in NumPy.  The previous KerasCV implementation
        # unconditionally used TensorFlow device kernels, which fails on
        # machines whose installed CUDA PTX does not match the GPU.
        img = np.asarray(NCHW_to_NHWC(img_nchw)).copy()
        rng = np.random.default_rng(seed)
        n, h, w, _ = img.shape
        cutout_h = max(1, int(round(h * height_factor)))
        cutout_w = max(1, int(round(w * width_factor)))
        for i in range(n):
            y = rng.integers(0, max(1, h - cutout_h + 1))
            x = rng.integers(0, max(1, w - cutout_w + 1))
            img[i, y : y + cutout_h, x : x + cutout_w, :] = 0
        return NHWC_to_NCHW(img)

    @staticmethod
    def random_cutout_left_side(
        img_nchw: Array,
        width_factor: float = 0.5,
        seed: int = 10,
    ):
        """Black out a contiguous left strip while keeping full height."""
        # Keep augmentation in NumPy to avoid GPU/PTX compatibility issues.
        img = np.asarray(NCHW_to_NHWC(img_nchw)).copy()
        _ = seed
        n, h, w, _ = img.shape
        cutout_w = max(1, int(round(w * np.clip(width_factor, 0.0, 1.0))))
        # Deterministic: always blacken the leftmost columns.
        img[:, :, :cutout_w, :] = 0
        return NHWC_to_NCHW(img)

    @staticmethod
    def random_cutout_circle(img_nchw: Array, seed: int):
        img = np.asarray(NCHW_to_NHWC(img_nchw)).copy()
        n, h, w, _ = img.shape
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[-1 : 1 : complex(0, h), -1 : 1 : complex(0, w)]
        for i in range(n):
            cx, cy = rng.uniform(-0.5, 0.5, size=2)
            radius = rng.uniform(0.05, 0.2)
            mask = ((xx - cx) / radius) ** 2 + ((yy - cy) / radius) ** 2 < 1.0
            img[i][mask] = 0
        return NHWC_to_NCHW(img)

    @staticmethod
    def random_cutout(
        img_nchw: Array,
        seed: int,
        strategies: tuple = ("circle",),
        square_height_factor_range: tuple = (0.05, 0.25),
        square_width_factor_range: tuple = (0.05, 0.25),
        left_width_factor_range: tuple = (0.25, 0.5),
        noise_probability: float = 0.0,
        noise_scale: tuple = (0.0, 0.2),
    ):
        """Apply a random cutout strategy per sample.

        Args:
            img_nchw: Input tensor in NCHW.
            seed: RNG seed.
            strategies: Iterable of strategies from
                ``{"circle", "ellipse", "square", "left_side", "random_side"}``.
            square_height_factor_range: (min,max) for square/circle height scaling.
            square_width_factor_range: (min,max) for square/circle width scaling.
            left_width_factor_range: (min,max) for the left-side strip width.
            noise_probability: Probability of adding RGB noise to the killed region.
                Alpha and hidden state channels always remain zero so damaged cells
                cannot survive through injected state.
            noise_scale: Optional ``(min, max)`` range for RGB noise in cutout.
        """
        img = np.asarray(NCHW_to_NHWC(img_nchw)).copy()
        rng = np.random.default_rng(seed)
        n, h, w, c = img.shape
        strategies = list(strategies) if strategies else ["circle"]

        def _sample_factor(value_range, lower=0.01, upper=1.0):
            if isinstance(value_range, (list, tuple)):
                lo, hi = float(value_range[0]), float(value_range[1])
                if lo > hi:
                    lo, hi = hi, lo
                value = rng.uniform(lo, hi)
            else:
                value = float(value_range)
            return float(np.clip(value, lower, upper))

        def _fill_region(region):
            fill = np.zeros(region.shape, dtype=img.dtype)
            if noise_probability > 0.0 and rng.random() < noise_probability:
                if isinstance(noise_scale, (list, tuple)):
                    min_scale = np.clip(float(noise_scale[0]), 0.0, 1.0)
                    max_scale = np.clip(float(noise_scale[1]), 0.0, 1.0)
                    if min_scale > max_scale:
                        min_scale, max_scale = max_scale, min_scale
                    scale = rng.uniform(min_scale, max_scale)
                else:
                    scale = np.clip(float(noise_scale), 0.0, 1.0)
                visible_channels = min(3, region.shape[-1])
                fill[..., :visible_channels] = (
                    rng.random(region[..., :visible_channels].shape) * scale
                )
            return fill

        for i in range(n):
            strategy = rng.choice(strategies)
            if strategy in {"square", "rect", "rectangle"}:
                hh = int(_sample_factor(square_height_factor_range) * h)
                ww = int(_sample_factor(square_width_factor_range) * w)
                hh = max(1, min(h, hh))
                ww = max(1, min(w, ww))
                y = rng.integers(0, max(1, h - hh + 1))
                x = rng.integers(0, max(1, w - ww + 1))
                img[i, y : y + hh, x : x + ww, :] = _fill_region(
                    img[i, y : y + hh, x : x + ww, :]
                )
            elif strategy in {
                "left",
                "left_side",
                "left-strip",
                "left_strip",
                "random_side",
                "side",
            }:
                side = (
                    str(rng.choice(("left", "right", "top", "bottom")))
                    if strategy in {"random_side", "side"}
                    else "left"
                )
                factor = _sample_factor(left_width_factor_range)
                if side in {"left", "right"}:
                    extent = max(1, int(round(w * factor)))
                    region = (
                        img[i, :, :extent, :]
                        if side == "left"
                        else img[i, :, w - extent :, :]
                    )
                else:
                    extent = max(1, int(round(h * factor)))
                    region = (
                        img[i, :extent, :, :]
                        if side == "top"
                        else img[i, h - extent :, :, :]
                    )
                region[...] = _fill_region(region)
            else:  # circle/ellipse fallback
                yy, xx = np.mgrid[0:h, 0:w]
                radius_y = max(
                    1.0,
                    0.5 * h * _sample_factor(square_height_factor_range, 0.02, 1.0),
                )
                radius_x = max(
                    1.0,
                    0.5 * w * _sample_factor(square_width_factor_range, 0.02, 1.0),
                )
                if strategy not in {"ellipse", "oval"}:
                    radius_x = radius_y = min(radius_x, radius_y)
                # Keep the center inside the canvas while allowing the cutout
                # to clip an edge; this produces partial and central injuries.
                cx = rng.uniform(0.0, max(1.0, w - 1.0))
                cy = rng.uniform(0.0, max(1.0, h - 1.0))
                angle = rng.uniform(0.0, np.pi) if strategy in {"ellipse", "oval"} else 0.0
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                dx, dy = xx - cx, yy - cy
                rotated_x = cos_a * dx + sin_a * dy
                rotated_y = -sin_a * dx + cos_a * dy
                mask = (rotated_x / radius_x) ** 2 + (
                    rotated_y / radius_y
                ) ** 2 < 1.0
                img[i][mask] = _fill_region(img[i][mask])

        return NHWC_to_NCHW(img)
