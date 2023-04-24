import pytest
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax  # type: ignore
from dataclasses import dataclass
import cv2  # type: ignore
import os
import numpy as np
from functools import partial

from nca.model import UpdateModel
from nca.trainer import (
    create_state,
    train_step,
    create_cell_update_fn,
    evaluate_step,
    train_and_evaluate,
    evaluate,
)
from nca.config import NCAConfig
from nca.utils import NCHW_to_NHWC, NHWC_to_NCHW


@pytest.fixture
def config():
    return NCAConfig(
        dimensions=(32, 32),
        model_output_len=16,
        batch_size=2,
        num_steps=2,
        eval_every=1,
        checkpoint_dir="/tmp/test_ckpt",
        checkpoint_every=1,
        validation_video_dir="/tmp/test_vid",
    )


@pytest.fixture
def dummy_state(config):
    return create_state(config)[0]


def test_create_state(config):
    state = create_state(config)[0]

    assert isinstance(state, train_state.TrainState)
    assert isinstance(state.apply_fn, UpdateModel)
    assert isinstance(state.tx, optax.GradientTransformation)


def test_cell_update_fn(config):
    key = jax.random.PRNGKey(0)
    state_grid = jnp.zeros((1,) + config.dimensions + (config.model_output_len,))
    state_grid = NHWC_to_NCWH(state_grid)
    model = UpdateModel(model_output_len=config.model_output_len)

    cell_update_fn = create_cell_update_fn(config, model)

    params = model.init(
        key,
        jax.random.normal(
            key,
            (1,) + config.dimensions + (config.model_output_len * 3,),
        ),
    )

    new_state_grid = cell_update_fn(key, state_grid, params)
    new_state_grid = NCHW_to_NHWC(new_state_grid)

    assert new_state_grid.shape == (1,) + config.dimensions + (config.model_output_len,)


def test_train_step(config, dummy_state):
    key = jax.random.PRNGKey(0)
    state_grid = jnp.zeros(
        (config.batch_size,) + config.dimensions + (config.model_output_len,)
    )
    target = jnp.zeros((config.batch_size,) + config.dimensions + (4,))

    cell_update_fn = create_cell_update_fn(config, dummy_state.apply_fn)

    state_grid = NHWC_to_NCWH(state_grid)
    target = NHWC_to_NCWH(target)

    state, loss, _, _ = train_step(
        key, dummy_state, state_grid, target, cell_update_fn, num_steps=14
    )

    assert loss.shape == ()


def test_eval_step(config, dummy_state):
    key = jax.random.PRNGKey(0)
    bs = 6
    state_grid = jnp.zeros((bs,) + config.dimensions + (config.model_output_len,))
    target = jnp.zeros((bs,) + config.dimensions + (4,))

    cell_update_fn = create_cell_update_fn(config, dummy_state.apply_fn)

    state_grid = NHWC_to_NCWH(state_grid)
    target = NHWC_to_NCWH(target)

    state_grids, loss = evaluate_step(
        dummy_state, state_grid, target, cell_update_fn, num_steps=11
    )
    _, loss_no_reduce = evaluate_step(
        dummy_state, state_grid, target, cell_update_fn, num_steps=11, reduce_loss=False
    )
    assert len(state_grids) == 11
    assert loss_no_reduce.shape == (bs, 4, 32, 32)
    assert loss.shape == ()


def test_train_and_evaluate(config):
    print("Starting train_and_evaluate test")
    train_and_evaluate(config)
    print("Finished train_and_evaluate test")

    # check if the checkpoint was saved
    os.path.exists(config.checkpoint_dir)

    # check if video was saved
    os.path.exists(config.validation_video_dir)

    # rm testing dirs
    os.system(f"rm -rf {config.checkpoint_dir}")
    os.system(f"rm -rf {config.validation_video_dir}")


def test_evaluation(config):
    evaluate(config)

    os.path.exists(config.validation_video_dir)

    os.system(f"rm -rf {config.validation_video_dir}")
