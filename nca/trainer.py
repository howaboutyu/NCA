import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state
import optax
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Callable
import tensorflow as tf
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter


from nca.model import UpdateModel
from nca.nca import create_perception_kernel, perceive, cell_update
from nca.config import NCAConfig
from nca.dataset import NCADataGenerator
from nca.utils import make_video, NCHW_to_NHWC


def create_state(config: NCAConfig) -> train_state.TrainState:
    learning_rate_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate, decay_steps=config.num_steps
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


def create_cell_update_fn(
    config: NCAConfig,
    model_fn: UpdateModel,
    use_jit: bool = True,
) -> Callable:
    # create perception kernels for updating the state grid
    kernel_x, kernel_y = create_perception_kernel(
        input_size=config.model_output_len,
        output_size=config.model_output_len,
        use_oihw_layout=True,
    )

    # define a function to update the cell state grid using the provided model function and parameters
    def cell_update_fn(key, state_grid, params):
        # call the cell_update function with the provided inputs and the perception kernels
        return cell_update(
            key=key,
            state_grid=state_grid,
            params=params,
            model_fn=model_fn,
            kernel_x=kernel_x,
            kernel_y=kernel_y,
            update_prob=0.5,
        )

    # if we want to use jit, then jit the cell_update_fn function
    if use_jit:
        cell_update_fn = jax.jit(cell_update_fn)

    # return the cell_update_fn function
    return cell_update_fn


def train_step(
    key: jnp.ndarray,
    state: train_state.TrainState,
    state_grid: jnp.ndarray,
    target: jnp.ndarray,
    cell_update_fn: Callable,
    num_steps: int = 64,
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

    def loss_fn(
        params: jnp.ndarray, state_grid: jnp.ndarray, key: jnp.ndarray
    ) -> jnp.ndarray:
        def body_fun(
            i: int, vals: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            key, state_grid = vals
            _, key = jax.random.split(key)
            state_grid = cell_update_fn(key, state_grid, params)
            return (key, state_grid)

        (key, state_grid) = jax.lax.fori_loop(0, num_steps, body_fun, (key, state_grid))

        pred_rgb = state_grid[:, :3]
        alpha = state_grid[:, 3:4]
        alpha = jnp.clip(alpha, 0, 1)
        pred_rgb = pred_rgb * alpha

        return jnp.mean(jnp.square(pred_rgb - target))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params, state_grid, key)
    state = state.apply_gradients(grads=grad)

    return state, loss


def evaluate_step(
    state: train_state.TrainState,
    state_grid: jnp.ndarray,
    target: jnp.ndarray,
    cell_update_fn: Callable,
    num_steps: int = 64,
    reduce_loss: bool = True,
) -> Tuple[List[jnp.ndarray], float, float]:
    """Runs a single evaluation step.

    Args:
        state: The current Flax training state.
        state_grid: The current cell state grid.
        target: The target RGB values.
        cell_update_fn: A function that updates the cell state grid using the provided model function and parameters.

    Returns:
        state_grids: A list of the cell state grids after each step. NCWH format.
        loss: The loss value for this step.
        SSIM: The SSIM value for this step.
    """

    # define a function that takes the model parameters and cell state grid as inputs and returns the predicted RGB values
    def predict_fn(params: jnp.ndarray, state_grid: jnp.ndarray) -> jnp.ndarray:
        state_grids = []
        key = jax.random.PRNGKey(0)
        for i in range(num_steps):
            _, key = jax.random.split(key)
            state_grid = cell_update_fn(key, state_grid, params)
            state_grids.append(state_grid)

        # extract the predicted RGB values and alpha channel from the state grid
        pred_rgb = state_grid[:, :3]
        alpha = state_grid[:, 3:4]
        # clip the alpha channel values to be between 0 and 1
        alpha = jnp.clip(alpha, 0, 1)
        pred_rgb = pred_rgb * alpha

        return pred_rgb, state_grids

    # call the predict_fn with the current state parameters and cell state grid to get the predicted RGB values
    pred_rgb, state_grids = predict_fn(state.params, state_grid)
    # convert the predicted RGB values and target to TensorFlow tensors
    pred_rgb_tf = tf.convert_to_tensor(pred_rgb)
    pred_rgb_tf = tf.transpose(pred_rgb_tf, perm=[0, 2, 3, 1])
    tf_target = tf.convert_to_tensor(target)
    tf_target = tf.transpose(tf_target, perm=[0, 2, 3, 1])

    # compute the SSIM score between the predicted and target images using TensorFlow
    pred_rgb_tf = tf.cast(pred_rgb_tf, tf.float32)
    tf_target = tf.cast(tf_target, tf.float32)
    ssim = tf.reduce_mean(tf.image.ssim(pred_rgb_tf, tf_target, max_val=1.0))
    # compute the loss between the predicted and target images using JAX
    if reduce_loss:
        loss = jnp.mean(jnp.square(pred_rgb - target))
    else:
        loss = jnp.square(pred_rgb - target)

    # return the predicted RGB values, loss, and SSIM score as a tuple
    return state_grids, loss, ssim


def train_and_evaluate(config: NCAConfig):
    """Runs the training and evaluation loop.

    Args:
        config: The NCAConfig object containing the training configuration.
    """

    state = create_state(config)

    if config.checkpoint_dir:
        state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)

    cell_update_fn = create_cell_update_fn(config, state.apply_fn)

    dataset_generator = NCADataGenerator(
        pool_size=config.pool_size,
        batch_size=config.batch_size,
        dimensions=config.dimensions,
        model_output_len=config.model_output_len,
    )

    train_target = dataset_generator.get_target(config.target_filename)

    # this is experiment 1 so we will use the same random key for all of the training
    data_key = jax.random.PRNGKey(0)

    # create a random key for generating subkeys
    key = jax.random.PRNGKey(0)

    tb_writer = SummaryWriter(config.log_dir)

    for step in range(state.step, config.num_steps):
        # get the training data
        state_grid, _ = dataset_generator.sample(data_key)

        # split the random key into two subkeys
        key, _ = jax.random.split(key)

        # create a random number between 64 and 96
        nca_steps = jax.random.randint(key, shape=(), minval=64, maxval=96)

        state, loss = train_step(
            key, state, state_grid, train_target, cell_update_fn, nca_steps
        )

        print(f"Loss : {loss}")
        tb_writer.add_scalar("loss", np.asarray(loss), state.step)

        if step % config.eval_every == 0:
            # get the seed
            seed_grid = dataset_generator.seed_state[np.newaxis, ...]

            val_state_grids, loss, ssim = evaluate_step(
                state, seed_grid, train_target[:1], cell_update_fn
            )

            tb_state_grids = np.array(val_state_grids)
            tb_state_grids = np.squeeze(tb_state_grids)[:, :3][np.newaxis, ...]
            tb_writer.add_video("val video", vid_tensor=tb_state_grids, fps=5)

            val_state_grids = [NCHW_to_NHWC(grid) for grid in val_state_grids]
            os.makedirs(config.validation_video_dir, exist_ok=True)
            output_video_file = os.path.join(config.validation_video_dir, f"{step}.mp4")
            make_video(val_state_grids, output_video_file)

        if step % config.checkpoint_every == 0 and config.checkpoint_dir:
            checkpoints.save_checkpoint(
                config.checkpoint_dir, state, step=state.step, keep=3
            )
