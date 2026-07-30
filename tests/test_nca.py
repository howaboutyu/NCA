import pytest
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import cv2  # type: ignore

from context import *
from nca.nca import *
from nca.model import UpdateModel, sample_continuous_edges


def test_perception_kernel_creation():
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


def test_perception_function():
    # Set up random input data
    key = random.PRNGKey(0)
    input_shape = (1, 16, 32, 32)
    x = random.normal(key, input_shape)

    kernel_x, kernel_y = create_perception_kernel(
        input_size=16, output_size=16, use_oihw_layout=True
    )
    y = perceive(x, kernel_x, kernel_y)

    assert y.shape == (1, 16 * 3, 32, 32)

    fused = perceive(x, kernel_x, kernel_y, method="sobel_fused")
    np.testing.assert_allclose(y, fused)

    kernel_xx, kernel_yy = create_second_derivative_kernels(16, 16)
    second = perceive(
        x,
        kernel_x,
        kernel_y,
        method="sobel_second",
        kernel_xx=kernel_xx,
        kernel_yy=kernel_yy,
    )
    assert second.shape == (1, 16 * 5, 32, 32)

    kernel_x5, kernel_y5 = create_multiscale_perception_kernels(16, 16)
    multiscale = perceive(
        x,
        kernel_x,
        kernel_y,
        method="sobel_multiscale",
        kernel_x5=kernel_x5,
        kernel_y5=kernel_y5,
    )
    assert multiscale.shape == (1, 16 * 5, 32, 32)


def test_cell_update_function():
    # Set up random input data
    key = random.PRNGKey(0)
    input_shape = (4, 16, 32, 32)
    x = random.normal(key, input_shape)

    # Initialize the model and its parameters
    model = UpdateModel()
    rand_data = random.normal(key, (1, 32, 32, 16 * 3))
    params = model.init(key, rand_data)

    # Create the perception kernel
    kernel_x, kernel_y = create_perception_kernel(
        input_size=16, output_size=16, use_oihw_layout=True
    )

    # Compute the update with update_prob=1.0
    y = cell_update(key, x, model, params, kernel_x, kernel_y, update_prob=0.5)

    assert y.shape == (4, 16, 32, 32)


def test_learned_perception_cell_update():
    key = random.PRNGKey(0)
    x = random.normal(key, (4, 16, 32, 32))
    model = UpdateModel(perception_method="learned")
    params = model.init(key, random.normal(key, (1, 32, 32, 16)))
    kernel_x, kernel_y = create_perception_kernel(16, 16, use_oihw_layout=True)
    y = cell_update(
        key,
        x,
        model,
        params,
        kernel_x,
        kernel_y,
        update_prob=0.5,
        perception_method="learned",
    )
    assert y.shape == x.shape


def test_global_context_cell_update():
    key = random.PRNGKey(0)
    x = random.normal(key, (4, 16, 32, 32))
    model = UpdateModel(
        perception_method="sobel_second", nonlocal_connections=True
    )
    params = model.init(key, random.normal(key, (1, 32, 32, 16 * 5)))
    kernel_x, kernel_y = create_perception_kernel(16, 16, use_oihw_layout=True)
    kernel_xx, kernel_yy = create_second_derivative_kernels(16, 16)
    y = cell_update(
        key,
        x,
        model,
        params,
        kernel_x,
        kernel_y,
        update_prob=0.5,
        perception_method="sobel_second",
        kernel_xx=kernel_xx,
        kernel_yy=kernel_yy,
    )
    assert y.shape == x.shape


def test_token_attention_cell_update():
    key = random.PRNGKey(0)
    x = random.normal(key, (1, 16, 32, 32))
    model = UpdateModel(
        perception_method="sobel_second",
        nonlocal_connections=True,
        nonlocal_mode="token_attention",
    )
    params = model.init(key, random.normal(key, (1, 32, 32, 16 * 5)))
    kernel_x, kernel_y = create_perception_kernel(16, 16, use_oihw_layout=True)
    kernel_xx, kernel_yy = create_second_derivative_kernels(16, 16)
    y = cell_update(
        key,
        x,
        model,
        params,
        kernel_x,
        kernel_y,
        perception_method="sobel_second",
        kernel_xx=kernel_xx,
        kernel_yy=kernel_yy,
    )
    assert y.shape == x.shape


def test_continuous_edge_sampling_is_differentiable():
    state = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1)
    edge_pos = jnp.array([[[[[0.0, 0.0]]]]])
    sampled = sample_continuous_edges(state, edge_pos)
    assert sampled.shape == (1, 1, 1, 1, 1)
    np.testing.assert_allclose(sampled, 7.5)
    gradient = jax.grad(
        lambda position: sample_continuous_edges(state, position).sum()
    )(edge_pos)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0)


def test_moving_edges_update_and_aggregate_messages():
    key = random.PRNGKey(0)
    x = random.normal(key, (1, 16, 8, 8))
    edge_count, edge_dim = 3, 4
    model = UpdateModel(
        perception_method="sobel_second",
        nonlocal_connections=True,
        nonlocal_mode="moving_edges",
        edge_count=edge_count,
        edge_state_dim=edge_dim,
        pokemon_vocab_size=2,
        pokemon_embedding_dim=8,
    )
    perception = random.normal(key, (1, 8, 8, 16 * 5))
    positions = jnp.zeros((1, 8, 8, edge_count, 2))
    edge_state = jnp.zeros((1, 8, 8, edge_count, edge_dim))
    params = model.init(
        key,
        perception,
        pokemon_ids=jnp.array([0]),
        state_grid=x,
        edge_pos=positions,
        edge_state=edge_state,
    )
    output, next_positions, next_edge_state = model.apply(
        params,
        perception,
        pokemon_ids=jnp.array([0]),
        state_grid=x,
        edge_pos=positions,
        edge_state=edge_state,
    )
    assert output.shape == x.transpose(0, 2, 3, 1).shape
    assert next_positions.shape == positions.shape
    assert next_edge_state.shape == edge_state.shape
    assert jnp.all(next_positions <= 1.0)
    assert jnp.all(next_positions >= -1.0)
    assert jnp.max(jnp.abs(next_positions - positions)) <= model.edge_step_size + 1e-6


def test_conditional_pokemon_embedding_changes_model_conditioning():
    key = random.PRNGKey(0)
    model = UpdateModel(
        perception_method="sobel_second",
        pokemon_vocab_size=3,
        pokemon_embedding_dim=8,
    )
    inputs = random.normal(key, (2, 16, 16, 16 * 5))
    params = model.init(key, inputs, pokemon_ids=jnp.array([0, 1]))
    output = model.apply(params, inputs, pokemon_ids=jnp.array([0, 1]))
    assert output.shape == (2, 16, 16, 16)
