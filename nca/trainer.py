import jax
import jax.numpy as jnp

from flax.training import checkpoints, train_state
import optax  # type: ignore
from dataclasses import dataclass
import cv2  # type: ignore
import numpy as np
from typing import Tuple, List, Dict, Any, Callable
import tensorflow as tf  # type: ignore
from tqdm import tqdm  # type: ignore
import os
from tensorboardX import SummaryWriter  # type: ignore
from functools import partial

from nca.model import UpdateModel
from nca.nca import create_perception_kernel, perceive, cell_update
from nca.config import NCAConfig
from nca.dataset import NCADataGenerator
from nca.utils import make_video, NCHW_to_NHWC, NCHW_to_NHWC, mse

Array = Any


def create_state(config: NCAConfig) -> Tuple[train_state.TrainState, Any]:
    # Create a cosine learning rate decay schedule
    learning_rate_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate, decay_steps=config.total_training_steps
    )

    # Create an Adam optimizer with the learning rate schedule
    optimizer = optax.adam(learning_rate=learning_rate_schedule)

    # Initialize the model with random weights
    model = UpdateModel(model_output_len=config.model_output_len)
    initial_params = jax.random.normal(
        jax.random.PRNGKey(0),
        (1, config.dimensions[0], config.dimensions[1], config.model_output_len * 3),
    )
    initial_state = model.init(jax.random.PRNGKey(0), initial_params)

    # Create a TrainState object to hold the model state and optimizer state
    state = train_state.TrainState.create(
        apply_fn=model,
        params=initial_state,
        tx=optimizer,
    )

    # Return the TrainState object and the learning rate schedule
    return state, learning_rate_schedule


def create_cell_update_fn(
    config: NCAConfig,
    model_fn: Callable[..., Any],
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


def nca_looper(
    key: jax.random.PRNGKeyArray,
    params: Any,
    state_grid: Array,
    num_nca_steps: int,
    cell_update_fn: Callable,
) -> Tuple[Array, Array]:
    state_grid_sequence = []
    for _ in range(num_nca_steps):
        _, key = jax.random.split(key)
        state_grid = cell_update_fn(key, state_grid, params)
        state_grid_sequence.append(state_grid)

    pred_rgba = state_grid[:, :4]

    return pred_rgba, jnp.asarray(state_grid_sequence)


def train_step(
    key: jax.random.PRNGKeyArray,
    state: train_state.TrainState,
    state_grid: Array,
    target: Array,
    cell_update_fn: Callable,
    num_nca_steps: Array = 64,
    apply_grad: bool = True,
) -> Tuple[train_state.TrainState, Array, Array, Array]:
    """Runs a single training step.

    Args:
        key: A random key used for generating subkeys.
        state: The current Flax training state.
        state_grid: The current cell state grid.
        target: The target RGB values.
        cell_update_fn: A function that updates the cell state grid using the provided model function and parameters.

    Returns:
        *TODO*
    """

    def loss_fn(
        params: jnp.ndarray, state_grid: jnp.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[Array, Tuple[Array, Array]]:
        # Returns loss reduced over batch and spatial dimensions and loss not reduced over batch and spatial dimensions

        pred_rgba, state_grid_sequence = nca_looper(
            key,
            params,
            state_grid,
            num_nca_steps=num_nca_steps,
            cell_update_fn=cell_update_fn,
        )

        # used for visualizing the state grid during training
        jnp_state_grid_sequence = jnp.asarray(state_grid_sequence)

        return mse(pred_rgba, target), jnp_state_grid_sequence

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, state_grid_sequence), grad = grad_fn(state.params, state_grid, key)

    if apply_grad:
        state = state.apply_gradients(grads=grad)

    return state, loss, state_grid_sequence


def evaluate_step(
    state: train_state.TrainState,
    state_grid: Array,
    target: Array,
    cell_update_fn: Callable,
    num_nca_steps: int = 64,
    reduce_loss: bool = True,
    key=jax.random.PRNGKey(0),
) -> Tuple[List[Array], Array]:
    """Runs a single evaluation step.

    Args:
        state: The current Flax training state.
        state_grid: The current cell state grid.
        target: The target RGB values.
        cell_update_fn: A function that updates the cell state grid using the provided model function and parameters.

    Returns:
        state_grids: A list of the cell state grids after each step. NCWH format.
        loss: The loss value for this step.
    """

    pred_rgba, state_grids = nca_looper(
        key,
        params=state.params,
        state_grid=state_grid,
        num_nca_steps=num_nca_steps,
        cell_update_fn=cell_update_fn,
    )

    loss_value = mse(pred_rgba, target, reduce_loss)

    # return the predicted RGB values, loss as a tuple
    return state_grids, loss_value


def train_and_evaluate(config: NCAConfig):
    """Runs the training and evaluation loop.

    Args:
        config: The NCAConfig object containing the training configuration.
    """

    state, learning_rate_schedule = create_state(config)

    if config.weights_dir:
        state = checkpoints.restore_checkpoint(config.weights_dir, state)

    cell_update_fn = create_cell_update_fn(config, state.apply_fn, use_jit=False)

    dataset_generator = NCADataGenerator(
        pool_size=config.pool_size,
        batch_size=config.batch_size,
        dimensions=config.dimensions,
        model_output_len=config.model_output_len,
    )

    train_target = dataset_generator.get_target(config.target_filename)

    # key for data generator
    data_key = jax.random.PRNGKey(0)

    # create a random key for generating subkeys
    key = jax.random.PRNGKey(0)

    tb_writer = SummaryWriter(config.log_dir)

    # create a partial function for the train_step function
    p_train_step = partial(
        train_step,
        cell_update_fn=cell_update_fn,
        num_nca_steps=config.num_nca_steps,
        apply_grad=True,
    )

    # jit the train_step function
    train_step_jit = jax.jit(p_train_step)

    for step in range(state.step, config.total_training_steps):
        # get the training data
        state_grids, state_grid_indices = dataset_generator.sample(
            data_key, config.damage
        )

        # TODO : move computing batch_id_worst to dataset generator
        loss_non_reduced = mse(state_grids[:, :4], train_target, reduce_mean=False)
        loss_per_batch = jnp.mean(loss_non_reduced, axis=(1, 2, 3))

        # get the batch_id with the maximum loss
        batch_id_worst = jnp.argmax(loss_per_batch)
        state_grids[batch_id_worst] = dataset_generator.seed_state

        (
            state,
            loss,
            training_grid_array,
        ) = train_step_jit(
            key,
            state,
            state_grids,
            train_target,
        )

        # replace the pool with final state grid
        final_training_grid = np.squeeze(training_grid_array[-1])
        dataset_generator.update_pool(state_grid_indices, final_training_grid)
        print(f"state_grid_indices: {state_grid_indices}")
        print(f"Step : {step}, loss : {loss}")

        if step % config.log_every == 0:
            training_grid_array = np.clip(training_grid_array, 0.0, 1.0)

            alpha = training_grid_array[:, :, 3:4]
            rgb = training_grid_array[:, :, :3]
            training_grid_array = alpha * rgb

            # training_grid_array has shape (T, N, C, H, W) but add_video needs (N, T, C, H, W)
            training_grid_array = np.transpose(training_grid_array, (1, 0, 2, 3, 4))
            tb_writer.add_video(
                "training_grid", training_grid_array, state.step, fps=10
            )

            tb_writer.add_scalar("loss", np.asarray(loss), state.step)

            lr = learning_rate_schedule(state.step)
            tb_writer.add_scalar("lr", np.asarray(lr), state.step)

        if step % config.eval_every == 0:
            # get the seed
            seed_grid = dataset_generator.seed_state[np.newaxis, ...]

            val_state_grids, loss = evaluate_step(
                state,
                seed_grid,
                train_target[:1],
                cell_update_fn,
                num_nca_steps=config.total_eval_steps,
            )

            tb_writer.add_scalar("val_loss", np.asarray(loss), state.step)
            tb_writer.add_image("target_img", np.asarray(train_target[0]), state.step)

            tb_state_grids = np.array(val_state_grids)
            tb_state_grids = np.clip(tb_state_grids, 0.0, 1.0)
            tb_state_grids = np.squeeze(tb_state_grids)
            alpha = tb_state_grids[:, 3:4] > 0.1
            tb_state_grids = alpha * tb_state_grids[:, :3]
            tb_state_grids = tb_state_grids[np.newaxis, ...]

            # write to tb
            tb_writer.add_video(
                f"val_video", vid_tensor=tb_state_grids, fps=30, global_step=state.step
            )

            val_state_grids = [NCHW_to_NHWC(grid) for grid in val_state_grids]
            os.makedirs(config.validation_video_dir, exist_ok=True)
            output_video_file = os.path.join(config.validation_video_dir, f"{step}.mp4")
            make_video(val_state_grids, output_video_file)

        if step % config.checkpoint_every == 0 and config.checkpoint_dir:
            checkpoints.save_checkpoint(
                config.checkpoint_dir, state, step=state.step, keep=3
            )

        key, _ = jax.random.split(key)
        data_key, _ = jax.random.split(data_key)


def evaluate(config: NCAConfig, output_video_path=None):
    state, _ = create_state(config)

    if config.weights_dir:
        state = checkpoints.restore_checkpoint(config.weights_dir, state)

    cell_update_fn = create_cell_update_fn(config, state.apply_fn)

    dataset_generator = NCADataGenerator(
        pool_size=config.pool_size,
        batch_size=config.batch_size,
        dimensions=config.dimensions,
        model_output_len=config.model_output_len,
    )

    nca_looper_fn = partial(
        nca_looper, cell_update_fn=cell_update_fn, num_nca_steps=config.num_nca_steps
    )

    nca_looper_fn = jax.jit(nca_looper_fn)  # type: ignore

    num_loops = int(config.total_eval_steps // config.num_nca_steps)

    state_grid = dataset_generator.seed_state[np.newaxis, ...]
    state_grid_cache = []

    key = jax.random.PRNGKey(0)
    for n in range(num_loops):
        _, state_grid_array = nca_looper_fn(key, state.params, state_grid)
        state_grid = state_grid_array[-1]
        state_grid_cache.append(jnp.squeeze(state_grid_array))

        state_grid = NCADataGenerator.random_cutout(state_grid, max_size=(16, 16))
        state_grid = NCADataGenerator.random_cutout(state_grid, max_size=(16, 16))

    state_grid_cache = jnp.array(state_grid_cache)  # type: ignore

    state_grid_cache = jnp.concatenate(state_grid_cache, axis=0)
    state_grid_cache = jnp.clip(state_grid_cache, 0.0, 1.0)

    # TODO: add path to config.
    # save the entire state_grid_cache as npy.
    np.save("/tmp/state_grid_cache.npy", state_grid_cache)

    rgba = np.asarray(state_grid_cache)[:, 0:4]

    rgb = rgba[:, :3] * rgba[:, 3:4]

    # NCHW -> NHWC
    rgb = jnp.transpose(rgb, (0, 2, 3, 1))

    # resize with tf
    rgb = tf.image.resize(
        rgb, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    ).numpy()

    rgb = (rgb * 255).astype(np.uint8)

    if output_video_path is None:
        make_video(rgb, config.evaluation_video_file)
    else:
        make_video(rgb, output_video_path)
