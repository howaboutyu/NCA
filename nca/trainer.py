import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state
import optax
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Callable

from nca.model import UpdateModel
from nca.nca import create_perception_kernel, perceive, cell_update
from nca.config import NCAConfig


def create_state(config: NCAConfig) -> train_state.TrainState:
    learning_rate_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.num_epochs * config.steps_per_epoch,
    )

    tx = optax.adam(learning_rate=learning_rate_schedule)

    model = UpdateModel(model_output_len=config.model_output_len)

    state = train_state.TrainState.create(
        apply_fn=model,
        params=model.init(
            jax.random.PRNGKey(0),
            jax.random.normal(
                jax.random.PRNGKey(0),
                (
                    1,
                    config.dimensions[0],
                    config.dimensions[1],
                    config.model_output_len * 3,
                ),
            ),
        ),
        tx=tx,
    )

    return state


def create_cell_update_fn(config: NCAConfig) -> Callable:
    # create perception kernels for updating the state grid
    kernel_x, kernel_y = create_perception_kernel(
        input_size=config.model_output_len,
        output_size=config.model_output_len,
        use_oihw_layout=True,
    )

    # define a function to update the cell state grid using the provided model function and parameters
    def cell_update_fn(key, state_grid, model_fn, params):
        # call the cell_update function with the provided inputs and the perception kernels
        return cell_update(
            key,
            state_grid,
            model_fn,
            params,
            kernel_x,
            kernel_y,
            update_prob=0.5,
        )

    # return the cell_update_fn function
    return cell_update_fn


def train_step(
    key: jnp.ndarray,
    state: train_state.TrainState,
    state_grid: jnp.ndarray,
    target: jnp.ndarray,
    cell_update_fn: Callable,
) -> Tuple[train_state.TrainState, float]:
    """Runs a single training step.

    Args:
        key: A random key used for generating subkeys.
        state: The current Flax training state.
        state_grid: The current cell state grid.
        target: The target RGB values.
        cell_update_fn: A function that updates the cell state grid using the provided model function and parameters.

    Returns:
        A tuple containing the updated Flax training state and the loss value for this step.
    """

    # define a loss function that takes the model parameters, cell state grid, and random key as inputs
    def loss_fn(
        params: jnp.ndarray, state_grid: jnp.ndarray, key: jnp.ndarray
    ) -> jnp.ndarray:
        # define a function to update the cell state grid for a fixed number of steps
        def body_fun(
            i: int, vals: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # unpack the current state key and grid from the input tuple
            key, state_grid = vals
            # split the key to generate a new subkey for each step
            _, key = jax.random.split(key)
            # call the cell_update_fn with the current inputs to update the state grid
            state_grid = cell_update_fn(key, state_grid, state.apply_fn, params)
            # return the updated key and state grid as a tuple
            return (key, state_grid)

        # run the body_fun function for a fixed number of iterations using a loop
        (key, state_grid) = jax.lax.fori_loop(0, 64, body_fun, (key, state_grid))

        # extract the predicted RGB values and alpha channel from the state grid
        pred_rgb = state_grid[:, :3]
        alpha = state_grid[:, 3:4]
        # clip the alpha channel values to be between 0 and 1
        alpha = jnp.clip(alpha, 0, 1)
        pred_rgb = pred_rgb * alpha

        return jnp.mean(jnp.square(pred_rgb - target))

    # define a function that computes the gradient of the loss function with respect to the model parameters
    grad_fn = jax.value_and_grad(loss_fn)
    # compute the loss and gradient using the current state parameters, cell state grid, and random key
    loss, grad = grad_fn(state.params, state_grid, key)
    # apply the computed gradients to the current state to update the parameters
    state = state.apply_gradients(grads=grad)

    # return the updated state and loss
    return state, loss
