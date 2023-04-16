import pytest
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from dataclasses import dataclass
import cv2
import numpy as np
from functools import partial

from nca.model import UpdateModel
from nca.trainer import create_state, train_step, create_cell_update_fn
from nca.config import NCAConfig
from nca.utils import NCWH_to_NHWC, NHWC_to_NCWH


@pytest.fixture
def config():
    return NCAConfig(dimensions=(32, 32), model_output_len=16, batch_size=1)


@pytest.fixture
def dummy_state(config):
    return create_state(config)


def test_create_state(config):
    state = create_state(config)

    assert isinstance(state, train_state.TrainState)
    assert isinstance(state.apply_fn, UpdateModel)
    assert isinstance(state.tx, optax.GradientTransformation)


def test_cell_update_fn(config):
    cell_update_fn = create_cell_update_fn(config)

    key = jax.random.PRNGKey(0)
    state_grid = jnp.zeros((1,) + config.dimensions + (config.model_output_len,))
    state_grid = NHWC_to_NCWH(state_grid)
    model = UpdateModel(model_output_len=config.model_output_len)
    params = model.init(
        key,
        jax.random.normal(
            key,
            (1,) + config.dimensions + (config.model_output_len * 3,),
        ),
    )
    print(model.apply)

    new_state_grid = cell_update_fn(key, state_grid, model, params)
    new_state_grid = NCWH_to_NHWC(new_state_grid)

    assert new_state_grid.shape == (1,) + config.dimensions + (config.model_output_len,)


def test_train_step(config, dummy_state):
    key = jax.random.PRNGKey(0)
    state_grid = jnp.zeros((1,) + config.dimensions + (config.model_output_len,))
    target = jnp.zeros((1,) + config.dimensions + (3,))

    cell_update_fn = create_cell_update_fn(config)

    state_grid = NHWC_to_NCWH(state_grid)
    target = NHWC_to_NCWH(target)

    train_step(key, dummy_state, state_grid, target, cell_update_fn)
