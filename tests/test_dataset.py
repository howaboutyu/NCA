import pytest
import jax
import numpy as np
from context import *
from nca.dataset import NCADataGenerator
from nca.utils import NCHW_to_NHWC, NHWC_to_NCHW


@pytest.fixture
def generator() -> NCADataGenerator:
    return NCADataGenerator(100, 32, (40, 40), 16)


def test_initialization(generator: NCADataGenerator):
    assert generator.pool.shape == (100, 16, 40, 40)


def test_random_seed_distribution():
    generator = NCADataGenerator(
        2, 1, (40, 40), 16, seed_density=0.2, seed_random_seed=0
    )
    alive = generator.seed_state[3] > 0
    assert 0.15 < alive.mean() < 0.25
    np.testing.assert_array_equal(alive, generator.seed_state[4] > 0)


def test_pokeball_seed_is_compact_and_colored():
    generator = NCADataGenerator(
        1, 1, (56, 56), 16, seed_pattern="pokeball", seed_size=11
    )
    alive = generator.seed_state[3] > 0
    assert alive.sum() < 150
    assert alive[28, 28]
    assert generator.seed_state[0, 25, 28] > generator.seed_state[1, 25, 28]
    assert np.all(generator.seed_state[3:, alive] == 1.0)


def test_multi_pokemon_pool_assigns_balanced_ids():
    generator = NCADataGenerator(
        5,
        2,
        (40, 40),
        16,
        pokemon_targets=("a.png", "b.png"),
    )
    np.testing.assert_array_equal(generator.pool_pokemon_ids, [0, 1, 0, 1, 0])


def test_sample_generation(generator: NCADataGenerator):
    key = jax.random.PRNGKey(0)

    pool1, indices1 = generator.sample(key)

    key, subkey = jax.random.split(key)

    pool2, indices2 = generator.sample(subkey)

    assert not np.array_equal(indices1, indices2)
    assert pool2.shape == (32, 16, 40, 40)


def test_pool_update(generator: NCADataGenerator):
    indices = np.array([0, 1, 2, 3])
    new_states = np.ones((4, 16, 40, 40))

    generator.update_pool(indices, new_states)


def test_target_retrieval(generator: NCADataGenerator):
    target = generator.get_target("emoji_imgs/skier.png")
    assert target.shape == (32, 4, 40, 40)


def test_random_cutouts(generator: NCADataGenerator):
    # Generate random data
    data_nhwc = np.zeros((generator.batch_size,) + generator.dimensions + (3,))
    data_nchw = NHWC_to_NCHW(data_nhwc)

    masked_data = generator.random_cutout_circle(data_nchw, seed=0)
    assert masked_data.shape == data_nchw.shape

    masked_data = generator.random_cutout_rect(data_nchw, seed=0)
    assert masked_data.shape == data_nchw.shape


def test_random_cutout_noise_keeps_damaged_cells_dead():
    data = np.ones((4, 8, 16, 16), dtype=np.float32)
    damaged = np.asarray(
        NCADataGenerator.random_cutout(
            data,
            seed=3,
            strategies=("left_side",),
            left_width_factor_range=(0.5, 0.5),
            noise_probability=1.0,
            noise_scale=(0.4, 0.4),
        )
    )

    # RGB may contain visual noise, but alpha and hidden state must be killed.
    assert np.any(damaged[:, :3, :, :8] > 0.0)
    assert np.all(damaged[:, 3:, :, :8] == 0.0)
    assert np.all(damaged[:, :, :, 8:] == 1.0)


def test_random_side_cutouts_vary_edge_and_always_add_rgb_noise():
    data = np.ones((1, 8, 32, 32), dtype=np.float32)
    damaged_edges = set()

    for seed in range(24):
        damaged = np.asarray(
            NCADataGenerator.random_cutout(
                data,
                seed=seed,
                strategies=("random_side",),
                left_width_factor_range=(0.25, 0.25),
                noise_probability=1.0,
                noise_scale=(0.3, 0.3),
            )
        )
        dead = damaged[0, 3] == 0.0
        if np.all(dead[:, :8]):
            damaged_edges.add("left")
        if np.all(dead[:, -8:]):
            damaged_edges.add("right")
        if np.all(dead[:8, :]):
            damaged_edges.add("top")
        if np.all(dead[-8:, :]):
            damaged_edges.add("bottom")

        assert np.any(damaged[0, :3, dead] > 0.0)
        assert np.all(damaged[0, 3:, dead] == 0.0)

    assert damaged_edges == {"left", "right", "top", "bottom"}


def test_ellipse_cutout_varies_geometry_and_keeps_state_dead():
    data = np.ones((1, 8, 32, 32), dtype=np.float32)
    masks = []
    for seed in range(4):
        damaged = np.asarray(
            NCADataGenerator.random_cutout(
                data,
                seed=seed,
                strategies=("ellipse",),
                square_height_factor_range=(0.2, 0.6),
                square_width_factor_range=(0.1, 0.5),
                noise_probability=1.0,
                noise_scale=(0.2, 0.6),
            )
        )
        dead = damaged[0, 3] == 0.0
        masks.append(dead)
        assert np.any(dead)
        assert np.any(damaged[0, :3, dead] > 0.0)
        assert np.all(damaged[0, 3:, dead] == 0.0)

    assert any(not np.array_equal(masks[0], mask) for mask in masks[1:])
