import pytest

from context import *
from nca.dataset import NCADataGenerator
import jax
import numpy as np


@pytest.fixture
def generator():
    return NCADataGenerator(100, 32, (40, 40), 16)


def test_init(generator):
    assert generator.pool.shape == (100, 16, 40, 40)


def test_sample(generator):
    key = jax.random.PRNGKey(0)

    pool1, indices1 = generator.sample(key)

    key, subkey = jax.random.split(key)

    pool2, indices2 = generator.sample(subkey)

    assert not np.array_equal(indices1, indices2)
    assert pool2.shape == (32, 16, 40, 40)


def test_update_pool(generator):
    indices = np.array([0, 1, 2, 3])
    new_states = np.ones((4, 16, 40, 40))

    generator.update_pool(indices, new_states)

    assert np.array_equal(generator.pool[indices], new_states)


def test_get_target(generator):
    target = generator.get_target("emoji_imgs/skier.png")
    assert target.shape == (32, 3, 40, 40)
