import jax
import jax.numpy as jnp


from flax.training import checkpoints, train_state
from flax import core

import optax  # type: ignore
from dataclasses import dataclass
import cv2  # type: ignore
import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional
import tensorflow as tf  # type: ignore
from tqdm import tqdm  # type: ignore
import os
from tensorboardX import SummaryWriter  # type: ignore
from functools import partial

from nca.model import UpdateModel
from nca.nca import create_perception_kernel, cell_update
from nca.config import NCAConfig
from nca.dataset import NCADataGenerator, XorDataGenerator
from nca.utils import make_video, NCHW_to_NHWC, mse

# define some types
Array = jax.Array
FrozenDict = core.FrozenDict[str, Any]


def create_state(config: NCAConfig) -> Tuple[train_state.TrainState, Any]:
    # Create a cosine learning rate decay schedule
    learning_rate_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate, decay_steps=config.total_training_steps
    )

    # Create an Adam optimizer with the learning rate schedule
    optimizer = optax.chain(
        optax.clip(0.5),
        optax.adamw(learning_rate=learning_rate_schedule),
    )

    # Initialize the model with random weights
    model = UpdateModel(model_output_len=config.model_output_len)
    dummy_data = jax.random.normal(
        jax.random.PRNGKey(0),
        (1, config.dimensions[0], config.dimensions[1], config.model_output_len * 5),
    )

    if config.weights_dir:
        restored_dict = checkpoints.restore_checkpoint(config.weights_dir, target=None)

    if restored_dict == None:
        params = model.init(jax.random.PRNGKey(0), dummy_data)
        print("Initializing params from scratch")
    else:
        print(f"Loading params from ckpt {config.weights_dir}")
        params = restored_dict["params"]
    # Create a TrainState object to hold the model state and optimizer state
    state = train_state.TrainState.create(
        apply_fn=model,
        params=params,
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
            update_prob=config.stochastic_update_prob,
            use_non_local_perceive=config.use_non_local_perceive,
        )

    # if we want to use jit, then jit the cell_update_fn function
    if use_jit:
        cell_update_fn = jax.jit(cell_update_fn)

    # return the cell_update_fn function
    return cell_update_fn


def nca_looper(
    key: Array,
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
    key: Array,
    state: train_state.TrainState,
    state_grid: Array,
    target: Array,
    cell_update_fn: Callable,
    num_nca_steps: int = 64,
    apply_grad: Optional[bool] = True,
) -> Tuple[train_state.TrainState, Array, Array]:
    """Runs a single training step.

    Args:
        key: A random key used for generating subkeys.
        state: The current Flax training state.
        state_grid: The current cell state grid.
        target: The target RGB values.
        cell_update_fn: A function that updates the cell state grid using the provided model function and parameters.

    Returns:
        A tuple of the new training state, the loss, and the entire state grid sequence with config.num_nca_steps elements.

    """

    def loss_fn(
        params: jnp.ndarray, state_grid: jnp.ndarray, key: Array
    ) -> Tuple[Array, Array]:
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
        pred_rgb1 = pred_rgba[:, :4 , 0:4, -4:]
        target_rgb1 = target[:, :4  , 0:4, -4:]
        pred_rgb2 = pred_rgba[:, :4, -4:, -4:]
        target_rgb2 = target[:, :4 , -4:, -4:]


        loss_value = jnp.mean((pred_rgb1 - target_rgb1) ** 2 + (pred_rgb2 - target_rgb2) ** 2)

        return loss_value, jnp_state_grid_sequence
        #return mse(pred_rgba, target), jnp_state_grid_sequence

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, state_grid_sequence), grad = grad_fn(state.params, state_grid, key)

    if apply_grad:
        grad = jax.tree_map(
            lambda g: jnp.nan_to_num(g / (jnp.linalg.norm(g) + 1e-8)), grad
        )
        state = state.apply_gradients(grads=grad)

    return state, loss, state_grid_sequence


def evaluate_step(
    state: train_state.TrainState,
    state_grid: np.ndarray,
    target: Array,
    cell_update_fn: Callable,
    num_nca_steps: int = 64,
    reduce_loss: bool = True,
    key=jax.random.PRNGKey(0),
) -> Tuple[Array, Array]:
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

    pred_rgb = pred_rgba[:, :3]
    target = target[:,:3]

    loss_value = mse(pred_rgb, target, reduce_loss)

    # return the predicted RGB values, loss as a tuple
    return state_grids, loss_value


def train_and_evaluate(config: NCAConfig):
    """Runs the training and evaluation loop.

    Args:
        config: The NCAConfig object containing the training configuration.
    """

    state, learning_rate_schedule = create_state(config)

    cell_update_fn = create_cell_update_fn(config, state.apply_fn, use_jit=False)

    #dataset_generator = NCADataGenerator(
    dataset_generator = XorDataGenerator(
        pool_size=config.pool_size,
        batch_size=config.batch_size,
        dimensions=config.dimensions,
        model_output_len=config.model_output_len,
    )

    #use_file_name = False
    #if use_file_name: 


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
        state_grids, state_grid_indices = dataset_generator.sample(key, damage=False)


        # FOR IMAGE: 
        # train_target = dataset_generator.get_target(config.target_filename)
        # FOR LOGIC: 
        train_target = dataset_generator.get_target(state_grid_indices)

        #import pdb; pdb.set_trace()

        loss_non_reduced_np = np.asarray(
            #mse(state_grids[:, :3], train_target[:, :3], reduce_mean=False)
            mse(state_grids[:, :3, :, -5:], train_target[:, :3, :, -5:], reduce_mean=False)
        )
        
        loss_per_batch_np = np.mean(loss_non_reduced_np, axis=(1, 2, 3))

        loss_rank = np.argsort(loss_per_batch_np)[::-1]

        # Rank from highest to lowest loss
        state_grids_ranked = state_grids[loss_rank]
        train_target = train_target[loss_rank]

        # set the worst performing batch to the seed state
        state_grids_ranked[:1], target_state = dataset_generator.get_seed_state()

        train_target[:1] = target_state

        if config.n_damage > 0:
            # replace best performing states (config.n_damage) grids with random cutouts
            state_grids_ranked[-config.n_damage :] = (
                NCADataGenerator.random_cutout_circle(
                    state_grids_ranked[-config.n_damage :],
                    int(key[0]),  # type: ignore
                )
            )

        # shuffle
        shuffled_idx = jax.random.permutation(
            key, jnp.arange(state_grids_ranked.shape[0])
        )
        shuffled_idx = np.asarray(shuffled_idx)
        state_grids_ranked = state_grids_ranked[shuffled_idx]

        (
            state,
            loss,
            training_grid_array,
        ) = train_step_jit(
            key,
            state,
            state_grids_ranked,
            train_target,
        )

        # replace the pool with final state grid
        final_training_grid = np.squeeze(training_grid_array[-1])
        #if np.random.randint(0, 100) > 98:
        #dataset_generator.update_pool(state_grid_indices[:1], final_training_grid[:1])
        dataset_generator.update_pool(state_grid_indices, final_training_grid)
        print(f"training_grid_array min: {jnp.min(training_grid_array)}")
        print(f"training_grid_array max: {jnp.max(training_grid_array)}")
        print(f"state_grid_indices: {state_grid_indices}")
        print(f"Step : {step}, loss : {loss}")

        if step % config.log_every == 0:
            # Log training grids as a gif and display using tensorboardX
            training_grid_array = np.clip(training_grid_array, 0.0, 1.0)
            alpha = training_grid_array[:, :, 3:4]
            rgb = training_grid_array[:, :, :3]
            training_grid_array = alpha * rgb

            # training_grid_array has shape (T, N, C, H, W) but `add_video` fn needs (N, T, C, H, W)
            training_grid_array = np.transpose(training_grid_array, (1, 0, 2, 3, 4))
            tb_writer.add_video(
                "training_grid", training_grid_array, state.step, fps=10
            )

            tb_writer.add_scalar("loss", np.asarray(loss), state.step)

            lr = learning_rate_schedule(state.step)
            print(f"Learning rate : {lr}")
            tb_writer.add_scalar("lr", np.asarray(lr), state.step)

        if step % config.eval_every == 0:
            # Evaluate the model starting with a seed state and propagate for `config.total_eval_steps` steps
            # The gif is also logged with tensorboardX
            seed_grid, target_state = dataset_generator.get_seed_state()
            seed_grid = jnp.expand_dims(seed_grid, 0)
            target_state = jnp.expand_dims(target_state, 0)
            #import pdb; pdb.set_trace()



            val_state_grids, loss = evaluate_step(
                state,
                seed_grid,
                target_state,
                cell_update_fn,
                num_nca_steps=config.total_eval_steps,
            )

            tb_writer.add_scalar("val_loss", np.asarray(loss), state.step)
            tb_writer.add_image("target_img", np.asarray(target_state[0, :3]).astype(np.float32), state.step)

            tb_state_grids = np.array(val_state_grids)
            tb_state_grids = np.clip(tb_state_grids, 0.0, 1.0)
            tb_state_grids = np.squeeze(tb_state_grids)
            alpha = tb_state_grids[:, 3:4] > 0.1
            tb_state_grids = alpha * tb_state_grids[:, :3]
            tb_state_grids = tb_state_grids[np.newaxis, ...]

            # write to tb
            tb_writer.add_video(
                "val_video", vid_tensor=tb_state_grids, fps=30, global_step=state.step
            )

            val_state_grids = [NCHW_to_NHWC(grid) for grid in val_state_grids]
            os.makedirs(config.validation_video_dir, exist_ok=True)
            output_video_file = os.path.join(config.validation_video_dir, f"{step}.mp4")
            make_video(val_state_grids, output_video_file)

        if step % config.checkpoint_every == 0 and config.checkpoint_dir:
            # save checkpoint
            checkpoints.save_checkpoint(
                config.checkpoint_dir, state, step=state.step, keep=3, overwrite=True 
            )

        # split the key for the next step
        key, _ = jax.random.split(key)


def evaluate(config: NCAConfig, output_video_path: Optional[str] = None) -> None:
    """This function evaluates the model for `config.total_eval_steps` steps starting with a seed state.
        The output is a video (mp4) of the NCA propagation.

    Args:
        config (NCAConfig): The config object.
        output_video_path (optional):
            Where to save the video path, (sometime like /abc/eval.mp4). Defaults to None.
            if none then the video is saved to `config.evaluation_video_file`
    """
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

    state_grid = dataset_generator.get_seed_state()[np.newaxis, ...]
    state_grid_cache = []

    key = jax.random.PRNGKey(0)
    for n in range(num_loops):
        _, state_grid_array = nca_looper_fn(key, state.params, state_grid)
        state_grid = state_grid_array[-1]
        state_grid_cache.append(jnp.squeeze(state_grid_array))

        ###  Some cutout strategies ###
        # 1) split the grid into two and make one half black
        #
        # state_grid = np.asarray(state_grid).copy()
        # state_grid[:, : ,  : , :state_grid.shape[2]//2] = 0.0

        # 2) make a random rectange cutouts
        state_grid = NCADataGenerator.random_cutout_rect(
            state_grid,
            height_factor=0.2,
            width_factor=0.2,
            seed=int(key[0]),  # type: ignore
        )

        key, _ = jax.random.split(key)

    state_grid_cache = jnp.array(state_grid_cache)  # type: ignore

    state_grid_cache = jnp.concatenate(state_grid_cache, axis=0)
    # save the entire state_grid_cache as npy.
    # TODO: add path to config.

    np.save("/tmp/state_grid_cache.npy", state_grid_cache)

    state_grid_cache = jnp.clip(state_grid_cache, 0.0, 1.0)

    rgba = np.asarray(state_grid_cache)[:, 0:4]

    rgb = rgba[:, :3] * rgba[:, 3:4]
    # NCHW -> NHWC
    rgb = jnp.transpose(rgb, (0, 2, 3, 1))

    # resize with tf
    rgb = tf.image.resize(
        rgb, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    ).numpy()

    if output_video_path is None:
        make_video(rgb, config.evaluation_video_file)
    else:
        make_video(rgb, output_video_path)
