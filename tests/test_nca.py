import pytest
import numpy as np
from jax import random
import cv2

from context import *
from nca.nca import *
from nca.model import UpdateModel


def test_create_perception_kernel():
    kernel_x, kernel_y = create_perception_kernel(
        input_size=64, output_size=8, use_oihw_layout=True
    )
    assert kernel_x.shape == (8, 64, 3, 3)
    assert kernel_y.shape == (8, 64, 3, 3)

    kernel_x, kernel_y = create_perception_kernel(
        input_size=1, output_size=1, use_oihw_layout=True
    )
    assert kernel_x.shape == (1, 1, 3, 3)
    assert kernel_y.shape == (1, 1, 3, 3)


def test_perceive():
    # Set up random input data
    key = random.PRNGKey(0)
    input_shape = (1, 16, 32, 32)
    x = random.normal(key, input_shape)

    kernel_x, kernel_y = create_perception_kernel(
        input_size=16, output_size=16, use_oihw_layout=True
    )
    y = perceive(x, kernel_x, kernel_y)

    assert y.shape == (1, 16 * 3, 32, 32)


def test_update():
    # Set up random input data
    key = jax.random.PRNGKey(0)
    input_shape = (4, 16, 32, 32)
    x = jax.random.normal(key, input_shape)

    # Initialize the model and its parameters
    model = UpdateModel()
    rand_data = jax.random.normal(key, (1, 32, 32, 16 * 3))
    params = model.init(key, rand_data)

    # Create the perception kernel
    kernel_x, kernel_y = create_perception_kernel(
        input_size=16, output_size=16, use_oihw_layout=True
    )

    # Compute the update with update_prob=1.0
    y = cell_update(key, x, model.apply, params, kernel_x, kernel_y, update_prob=0.5)

    assert y.shape == (4, 16, 32, 32)
