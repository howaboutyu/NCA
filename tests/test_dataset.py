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


def test_random_cutout_circle(generator: NCADataGenerator):
    # Generate random data
    data_nhwc = np.zeros((generator.batch_size,) + generator.dimensions + (3,))
    data_nchw = NHWC_to_NCHW(data_nhwc)

    masked_data = generator.random_cutout_circle(data_nchw, seed=0)

    assert masked_data.shape == data_nchw.shape
