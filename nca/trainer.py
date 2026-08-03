import jax
import jax.numpy as jnp


from flax.training import checkpoints, train_state
from flax import core
from flax.traverse_util import flatten_dict, unflatten_dict

import optax  # type: ignore
from dataclasses import dataclass
import cv2  # type: ignore
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any, Callable, Optional
from tqdm import tqdm  # type: ignore
import os
from tensorboardX import SummaryWriter  # type: ignore
from functools import partial

from nca.model import UpdateModel
from nca.nca import (
    create_perception_kernel,
    create_second_derivative_kernels,
    create_multiscale_perception_kernels,
    cell_update,
    alive_masking,
)
from nca.config import NCAConfig
from nca.dataset import NCADataGenerator
from nca.utils import make_video, NCHW_to_NHWC, mse

# define some types
Array = jax.Array
FrozenDict = core.FrozenDict[str, Any]


def migrate_moving_edge_params(
    initialized_params: Any,
    restored_params: Any,
    pokemon_embedding_dim: int,
) -> Any:
    """Warm-start a moving-edge model from a local-only checkpoint.

    Matching tensors are copied directly. ``conv_1`` needs special handling
    because received synapse state is inserted between perception features and
    Pokémon conditioning. The new synapse input block remains zero-initialized,
    preserving the old model's behavior at the start of fine-tuning.
    """
    initialized_flat = flatten_dict(core.unfreeze(initialized_params))
    restored_flat = flatten_dict(core.unfreeze(restored_params))

    for path, restored_value in restored_flat.items():
        if (
            path in initialized_flat
            and initialized_flat[path].shape == restored_value.shape
        ):
            initialized_flat[path] = restored_value

    kernel_path = ("params", "conv_1", "kernel")
    if kernel_path in initialized_flat and kernel_path in restored_flat:
        new_kernel = initialized_flat[kernel_path]
        old_kernel = restored_flat[kernel_path]
        if new_kernel.shape[2] > old_kernel.shape[2]:
            conditioning_width = min(
                pokemon_embedding_dim,
                old_kernel.shape[2],
            )
            perception_width = old_kernel.shape[2] - conditioning_width
            new_kernel = new_kernel.at[:, :, :perception_width, :].set(
                old_kernel[:, :, :perception_width, :]
            )
            new_kernel = new_kernel.at[
                :, :, perception_width : new_kernel.shape[2] - conditioning_width, :
            ].set(0.0)
            if conditioning_width:
                new_kernel = new_kernel.at[:, :, -conditioning_width:, :].set(
                    old_kernel[:, :, -conditioning_width:, :]
                )
            initialized_flat[kernel_path] = new_kernel

    embedding_path = ("params", "pokemon_embedding", "embedding")
    if embedding_path in initialized_flat and embedding_path in restored_flat:
        new_embedding = initialized_flat[embedding_path]
        old_embedding = restored_flat[embedding_path]
        if new_embedding.shape[1:] == old_embedding.shape[1:]:
            copied_rows = min(new_embedding.shape[0], old_embedding.shape[0])
            initialized_flat[embedding_path] = new_embedding.at[:copied_rows].set(
                old_embedding[:copied_rows]
            )

    return core.freeze(unflatten_dict(initialized_flat))


def create_state(config: NCAConfig) -> Tuple[train_state.TrainState, Any]:
    # Create a cosine learning rate decay schedule
    learning_rate_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate, decay_steps=config.total_training_steps
    )

    # Create an Adam optimizer with the learning rate schedule
    optimizer = optax.chain(
        optax.clip(0.5),
        optax.adam(learning_rate=learning_rate_schedule),
    )

    # Initialize the model with random weights
    valid_methods = {
        "sobel",
        "sobel_fused",
        "sobel_second",
        "sobel_multiscale",
        "learned",
    }
    if config.perception_method not in valid_methods:
        raise ValueError(
            f"perception_method must be one of {sorted(valid_methods)}, "
            f"got {config.perception_method!r}"
        )
    if (
        config.nonlocal_connections
        and config.nonlocal_mode != "moving_edges"
        and config.perception_method != "sobel_second"
    ):
        raise ValueError(
            "nonlocal_connections is currently supported only with "
            "perception_method='sobel_second' or nonlocal_mode='moving_edges'"
        )
    model = UpdateModel(
        model_output_len=config.model_output_len,
        perception_method=config.perception_method,
        nonlocal_connections=config.nonlocal_connections,
        nonlocal_mode=config.nonlocal_mode,
        nonlocal_token_grid=config.nonlocal_token_grid,
        nonlocal_attention_dim=config.nonlocal_attention_dim,
        edge_count=config.edge_count,
        edge_state_dim=config.edge_state_dim,
        edge_step_size=config.edge_step_size,
        edge_state_step_size=config.edge_state_step_size,
        edge_message_scale=config.edge_message_scale,
        pokemon_vocab_size=len(config.pokemon_targets),
        pokemon_embedding_dim=config.pokemon_embedding_dim,
    )
    dummy_data = jax.random.normal(
        jax.random.PRNGKey(0),
        (
            1,
            config.dimensions[0],
            config.dimensions[1],
            config.model_output_len
            if config.perception_method == "learned"
            else config.model_output_len
            * (
                5
                if config.perception_method
                in {"sobel_second", "sobel_multiscale"}
                else 3
            ),
        ),
    )

    restored_dict = None
    if config.weights_dir:
        restored_dict = checkpoints.restore_checkpoint(
            os.path.abspath(config.weights_dir), target=None
        )

    init_kwargs = {
        "pokemon_ids": jnp.zeros((1,), dtype=jnp.int32),
    }
    if config.nonlocal_connections and config.nonlocal_mode == "moving_edges":
        init_kwargs.update(
            state_grid=jnp.zeros(
                (1, config.model_output_len, *config.dimensions),
                dtype=dummy_data.dtype,
            ),
            edge_pos=jnp.zeros(
                (1, *config.dimensions, config.edge_count, 2),
                dtype=dummy_data.dtype,
            ),
            edge_state=jnp.zeros(
                (1, *config.dimensions, config.edge_count, config.edge_state_dim),
                dtype=dummy_data.dtype,
            ),
        )
    initialized_params = model.init(
        jax.random.PRNGKey(0),
        dummy_data,
        **init_kwargs,
    )

    if restored_dict == None:
        params = initialized_params
        print("Initializing params from scratch")
    else:
        print(f"Loading params from ckpt {config.weights_dir}")
        restored_params = restored_dict["params"]
        initialized_flat = flatten_dict(core.unfreeze(initialized_params))
        restored_flat = flatten_dict(core.unfreeze(restored_params))
        has_shape_mismatch = any(
            path in initialized_flat
            and initialized_flat[path].shape != restored_value.shape
            for path, restored_value in restored_flat.items()
        )
        if (
            has_shape_mismatch
            or (
                config.nonlocal_connections
                and config.nonlocal_mode == "moving_edges"
                and "edge_net" not in restored_params.get("params", {})
            )
        ):
            params = migrate_moving_edge_params(
                initialized_params,
                restored_params,
                config.pokemon_embedding_dim if config.pokemon_targets else 0,
            )
            print("Migrated checkpoint parameters to the current model shape")
        else:
            params = restored_params

    # Create a TrainState object to hold the model state and optimizer state
    state = train_state.TrainState.create(
        apply_fn=model,
        params=params,
        tx=optimizer,
    )

    if config.checkpoint_dir:
        state = checkpoints.restore_checkpoint(
            os.path.abspath(config.checkpoint_dir), target=state
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
    kernel_xx = kernel_yy = None
    if config.perception_method == "sobel_second":
        kernel_xx, kernel_yy = create_second_derivative_kernels(
            input_size=config.model_output_len,
            output_size=config.model_output_len,
            use_oihw_layout=True,
        )
    kernel_x5 = kernel_y5 = None
    if config.perception_method == "sobel_multiscale":
        kernel_x5, kernel_y5 = create_multiscale_perception_kernels(
            input_size=config.model_output_len,
            output_size=config.model_output_len,
            use_oihw_layout=True,
        )

    # define a function to update the cell state grid using the provided model function and parameters
    def cell_update_fn(
        key,
        state_grid,
        params,
        pokemon_ids=None,
        owner_alive=None,
        edge_pos=None,
        edge_state=None,
    ):
        # call the cell_update function with the provided inputs and the perception kernels
        return cell_update(
            key=key,
            state_grid=state_grid,
            params=params,
            model_fn=model_fn,
            kernel_x=kernel_x,
            kernel_y=kernel_y,
            update_prob=config.stochastic_update_prob,
            perception_method=config.perception_method,
            kernel_xx=kernel_xx,
            kernel_yy=kernel_yy,
            kernel_x5=kernel_x5,
            kernel_y5=kernel_y5,
            pokemon_ids=pokemon_ids,
            state_clip=config.state_clip,
            owner_alive=owner_alive,
            edge_pos=edge_pos,
            edge_state=edge_state,
        )

    # if we want to use jit, then jit the cell_update_fn function
    if use_jit:
        cell_update_fn = jax.jit(cell_update_fn)

    return cell_update_fn


def nca_looper(
    key: Array,
    params: Any,
    state_grid: Array,
    num_nca_steps: int,
    cell_update_fn: Callable,
    pokemon_ids: Optional[Array] = None,
    edge_count: int = 0,
    edge_state_dim: int = 0,
    return_edge_history: bool = False,
) -> Tuple[Array, Array]:
    edge_pos = None
    edge_state = None
    if edge_count > 0:
        init_key, key = jax.random.split(key)
        batch, _, height, width = state_grid.shape
        edge_pos = jax.random.uniform(
            init_key,
            (batch, height, width, edge_count, 2),
            minval=-1.0,
            maxval=1.0,
        )
        edge_state = jnp.zeros(
            (batch, height, width, edge_count, edge_state_dim),
            dtype=jnp.float32,
        )
    state_grid_sequence = []
    edge_position_sequence = []
    for _ in range(num_nca_steps):
        _, key = jax.random.split(key)
        owner_alive = alive_masking(state_grid[:, 3, :, :])
        result = cell_update_fn(
            key,
            state_grid,
            params,
            pokemon_ids,
            owner_alive,
            edge_pos,
            edge_state,
        )
        if edge_count > 0:
            state_grid, edge_pos, edge_state = result
        else:
            state_grid = result
        state_grid_sequence.append(state_grid)
        if edge_count > 0 and return_edge_history:
            edge_position_sequence.append(edge_pos)

    pred_rgba = state_grid[:, :4]

    state_grid_sequence = jnp.asarray(state_grid_sequence)
    if edge_count > 0 and return_edge_history:
        return pred_rgba, state_grid_sequence, jnp.asarray(edge_position_sequence)
    return pred_rgba, state_grid_sequence


def train_step(
    key: Array,
    state: train_state.TrainState,
    state_grid: Array,
    target: Array,
    cell_update_fn: Callable,
    num_nca_steps: int = 64,
    apply_grad: Optional[bool] = True,
    pokemon_ids: Optional[Array] = None,
    edge_count: int = 0,
    edge_state_dim: int = 0,
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
            pokemon_ids=pokemon_ids,
            edge_count=edge_count,
            edge_state_dim=edge_state_dim,
        )

        # used for visualizing the state grid during training
        jnp_state_grid_sequence = jnp.asarray(state_grid_sequence)

        return mse(pred_rgba, target), jnp_state_grid_sequence

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, state_grid_sequence), grad = grad_fn(state.params, state_grid, key)

    if apply_grad:
        grad = jax.tree_util.tree_map(
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
    pokemon_ids: Optional[Array] = None,
    edge_count: int = 0,
    edge_state_dim: int = 0,
    return_edge_history: bool = False,
) -> Tuple[Array, Array, Optional[Array]]:
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

    result = nca_looper(
        key,
        params=state.params,
        state_grid=state_grid,
        num_nca_steps=num_nca_steps,
        cell_update_fn=cell_update_fn,
        pokemon_ids=pokemon_ids,
        edge_count=edge_count,
        edge_state_dim=edge_state_dim,
        return_edge_history=return_edge_history,
    )
    if return_edge_history:
        pred_rgba, state_grids, edge_positions = result
    else:
        pred_rgba, state_grids = result
        edge_positions = None

    loss_value = mse(pred_rgba, target, reduce_loss)

    # return the predicted RGB values, loss as a tuple
    if return_edge_history:
        return state_grids, loss_value, edge_positions
    return state_grids, loss_value


def sample_evaluation_pokemon_id(config: NCAConfig, key: Array) -> int:
    """Select one configured conditional target for a validation rollout."""
    target_count = len(config.pokemon_targets)
    if target_count <= 1 or not config.eval_random_pokemon_id:
        return 0
    return int(jax.random.randint(key, (), 0, target_count))


def evaluate_damage_rollout(
    config: NCAConfig,
    state: train_state.TrainState,
    state_grid: Array,
    cell_update_fn: Callable,
    pokemon_ids: Optional[Array] = None,
    key: Array = jax.random.PRNGKey(0),
    return_edge_history: bool = False,
) -> Tuple[Array, Optional[Array]]:
    """Run evaluation while applying scheduled randomized cutouts.

    Unlike restarting ``nca_looper`` for each segment, this preserves moving
    synapse positions and state across every damage event.
    """
    edge_pos = None
    edge_state = None
    edge_count = (
        config.edge_count
        if config.nonlocal_connections and config.nonlocal_mode == "moving_edges"
        else 0
    )
    if edge_count > 0:
        init_key, key = jax.random.split(key)
        batch, _, height, width = state_grid.shape
        edge_pos = jax.random.uniform(
            init_key,
            (batch, height, width, edge_count, 2),
            minval=-1.0,
            maxval=1.0,
        )
        edge_state = jnp.zeros(
            (batch, height, width, edge_count, config.edge_state_dim),
            dtype=jnp.float32,
        )

    start = max(0, int(config.inference_cutout_start_step))
    interval = int(config.inference_cutout_interval_steps)
    frames = []
    edge_frames = []
    for step in range(config.total_eval_steps):
        should_damage = (
            config.inference_apply_cutout
            and interval > 0
            and step >= start
            and (step - start) % interval == 0
        )
        if should_damage:
            state_grid = NCADataGenerator.random_cutout(
                state_grid,
                seed=step,
                strategies=config.inference_cutout_strategies,
                square_height_factor_range=config.inference_cutout_square_height_factor_range,
                square_width_factor_range=config.inference_cutout_square_width_factor_range,
                left_width_factor_range=config.inference_cutout_left_width_factor_range,
                noise_probability=config.inference_cutout_noise_probability,
                noise_scale=config.inference_cutout_noise_scale,
            )

        _, key = jax.random.split(key)
        owner_alive = alive_masking(state_grid[:, 3, :, :])
        result = cell_update_fn(
            key,
            state_grid,
            state.params,
            pokemon_ids,
            owner_alive,
            edge_pos,
            edge_state,
        )
        if edge_count > 0:
            state_grid, edge_pos, edge_state = result
            if return_edge_history:
                edge_frames.append(edge_pos)
        else:
            state_grid = result
        frames.append(state_grid)

    edge_history = jnp.asarray(edge_frames) if edge_frames else None
    return jnp.asarray(frames), edge_history


def make_connection_overlay_video(
    images: list[np.ndarray],
    edge_positions: np.ndarray,
    filename: str,
    cell_stride: int = 4,
    fps: int = 10,
) -> np.ndarray:
    """Render evolving moving edges over an NHWC evaluation sequence."""
    rendered = []
    positions = np.asarray(edge_positions)[:, 0]
    for image, frame_positions in zip(images, positions):
        image = np.squeeze(np.asarray(image))
        base = np.asarray(image[..., :3] * 255.0, dtype=np.uint8)
        height, width = base.shape[:2]
        scale = 4
        canvas = cv2.resize(
            cv2.cvtColor(base, cv2.COLOR_RGB2BGR),
            (width * scale, height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = (canvas.astype(np.float32) * 0.25).astype(np.uint8)
        overlay = canvas.copy()
        for y in range(0, height, cell_stride):
            for x in range(0, width, cell_stride):
                if image.shape[-1] >= 4 and image[y, x, 3] <= 0.1:
                    continue
                origin = (x * scale, y * scale)
                for edge_index, position in enumerate(frame_positions[y, x]):
                    destination_x = int(
                        np.clip((position[0] + 1.0) * (width - 1) / 2.0, 0, width - 1)
                        * scale
                    )
                    destination_y = int(
                        np.clip((position[1] + 1.0) * (height - 1) / 2.0, 0, height - 1)
                        * scale
                    )
                    color = (
                        int(80 + 140 * ((edge_index * 67) % 255) / 255),
                        int(80 + 140 * ((edge_index * 131) % 255) / 255),
                        255,
                    )
                    cv2.line(
                        overlay,
                        origin,
                        (destination_x, destination_y),
                        color,
                        2,
                        cv2.LINE_AA,
                    )
        rendered.append(
            cv2.cvtColor(
                cv2.addWeighted(canvas, 0.35, overlay, 0.65, 0.0),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)
            / 255.0
        )
    make_video(rendered, filename, fps=fps)
    gif_filename = os.path.splitext(filename)[0] + ".gif"
    gif_frames = [
        Image.fromarray(np.asarray(frame * 255.0, dtype=np.uint8))
        for frame in rendered
    ]
    if gif_frames:
        gif_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=gif_frames[1:],
            duration=max(1, int(1000 / fps)),
            loop=0,
            optimize=False,
        )
    return np.asarray(rendered)


def train_and_evaluate(config: NCAConfig):
    """Runs the training and evaluation loop.

    Args:
        config: The NCAConfig object containing the training configuration.
    """

    state, learning_rate_schedule = create_state(config)

    cell_update_fn = create_cell_update_fn(config, state.apply_fn, use_jit=False)

    dataset_generator = NCADataGenerator(
        pool_size=config.pool_size,
        batch_size=config.batch_size,
        dimensions=config.dimensions,
        model_output_len=config.model_output_len,
        seed_density=config.seed_density,
        seed_noise_density=config.seed_noise_density,
        seed_random_seed=config.seed_random_seed,
        seed_pattern=config.seed_pattern,
        seed_size=config.seed_size,
        pokemon_targets=config.pokemon_targets,
    )

    targets_by_pokemon = dataset_generator.get_targets(config.target_filename)

    # create a random key for generating subkeys
    key = jax.random.PRNGKey(0)

    tb_writer = SummaryWriter(config.log_dir)

    # create a partial function for the train_step function
    p_train_step = partial(
        train_step,
        cell_update_fn=cell_update_fn,
        num_nca_steps=config.num_nca_steps,
        apply_grad=True,
        edge_count=(
            config.edge_count
            if config.nonlocal_connections and config.nonlocal_mode == "moving_edges"
            else 0
        ),
        edge_state_dim=config.edge_state_dim,
    )

    # jit the train_step function
    train_step_jit = jax.jit(p_train_step)

    for step in range(state.step, config.total_training_steps):
        key, sample_key, permute_key, cutoff_key, train_key, eval_key = jax.random.split(key, 6)

        # get the training data
        state_grids, state_grid_indices = dataset_generator.sample(sample_key, damage=False)
        pokemon_ids = dataset_generator.pool_pokemon_ids[state_grid_indices]
        batch_indices = np.arange(config.batch_size)
        train_target = targets_by_pokemon[pokemon_ids, batch_indices]

        loss_non_reduced_np = np.asarray(
            mse(state_grids[:, :4], train_target, reduce_mean=False)
        )
        loss_per_batch_np = np.mean(loss_non_reduced_np, axis=(1, 2, 3))

        loss_rank = np.argsort(loss_per_batch_np)[::-1]

        # Rank from highest to lowest loss
        state_grids_ranked = state_grids[loss_rank]
        target_ranked = train_target[loss_rank]
        pokemon_ids_ranked = pokemon_ids[loss_rank]
        pool_indices_ranked = state_grid_indices[loss_rank]

        # set the worst performing batch to the seed state
        state_grids_ranked[:1] = dataset_generator.seed_state

        if config.n_damage > 0:
            # replace best performing states (config.n_damage) grids with random cutouts
            state_grids_ranked[-config.n_damage :] = (
                NCADataGenerator.random_cutout(
                    state_grids_ranked[-config.n_damage :],
                    int(cutoff_key[0]),  # type: ignore
                    strategies=config.train_cutout_strategies,
                    square_height_factor_range=config.train_cutout_square_height_factor_range,
                    square_width_factor_range=config.train_cutout_square_width_factor_range,
                    left_width_factor_range=config.train_cutout_left_width_factor_range,
                    noise_probability=config.train_cutout_noise_probability,
                    noise_scale=config.train_cutout_noise_scale,
                )
            )

        # shuffle
        shuffled_idx = jax.random.permutation(
            permute_key, jnp.arange(state_grids_ranked.shape[0])
        )
        shuffled_idx = np.asarray(shuffled_idx)
        state_grids_ranked = state_grids_ranked[shuffled_idx]
        target_ranked = target_ranked[shuffled_idx]
        pokemon_ids_ranked = pokemon_ids_ranked[shuffled_idx]
        pool_indices_shuffled = pool_indices_ranked[shuffled_idx]

        (
            state,
            loss,
            training_grid_array,
        ) = train_step_jit(
            train_key,
            state,
            state_grids_ranked,
            target_ranked,
            pokemon_ids=pokemon_ids_ranked,
        )

        # replace the pool with final state grid
        final_training_grid = np.squeeze(training_grid_array[-1])
        dataset_generator.update_pool(pool_indices_shuffled, final_training_grid)
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
            seed_grid = dataset_generator.seed_state[np.newaxis, ...]
            eval_pokemon_id = sample_evaluation_pokemon_id(config, eval_key)
            eval_pokemon_ids = jnp.array([eval_pokemon_id], dtype=jnp.int32)

            evaluation_result = evaluate_step(
                state,
                seed_grid,
                targets_by_pokemon[eval_pokemon_id : eval_pokemon_id + 1],
                cell_update_fn,
                num_nca_steps=config.total_eval_steps,
                pokemon_ids=eval_pokemon_ids,
                edge_count=(
                    config.edge_count
                    if config.nonlocal_connections
                    and config.nonlocal_mode == "moving_edges"
                    else 0
                ),
                edge_state_dim=config.edge_state_dim,
                return_edge_history=(
                    config.nonlocal_connections
                    and config.nonlocal_mode == "moving_edges"
                ),
            )
            if config.nonlocal_connections and config.nonlocal_mode == "moving_edges":
                val_state_grids, loss, edge_positions = evaluation_result
            else:
                val_state_grids, loss = evaluation_result
                edge_positions = None

            tb_writer.add_scalar("val_loss", np.asarray(loss), state.step)
            tb_writer.add_scalar("val_pokemon_id", eval_pokemon_id, state.step)
            tb_writer.add_image(
                "target_img", np.asarray(targets_by_pokemon[eval_pokemon_id, 0]), state.step
            )

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

            val_state_grids = [
                np.clip(np.asarray(NCHW_to_NHWC(grid)), 0.0, 1.0)
                for grid in val_state_grids
            ]
            os.makedirs(config.validation_video_dir, exist_ok=True)
            output_video_file = os.path.join(config.validation_video_dir, f"{step}.mp4")
            make_video(val_state_grids, output_video_file)

            if config.inference_apply_cutout:
                damage_state_grids, _ = evaluate_damage_rollout(
                    config=config,
                    state=state,
                    state_grid=seed_grid,
                    cell_update_fn=cell_update_fn,
                    pokemon_ids=eval_pokemon_ids,
                )
                damage_video = np.clip(
                    np.squeeze(np.asarray(damage_state_grids)), 0.0, 1.0
                )
                damage_alpha = damage_video[:, 3:4] > 0.1
                damage_rgb = damage_alpha * damage_video[:, :3]
                tb_writer.add_video(
                    "val_damage_video",
                    vid_tensor=damage_rgb[np.newaxis, ...],
                    fps=30,
                    global_step=state.step,
                )
                damage_video_file = os.path.join(
                    config.validation_video_dir, f"{step}_damage.mp4"
                )
                make_video(
                    np.transpose(damage_rgb, (0, 2, 3, 1)),
                    damage_video_file,
                )

            if edge_positions is not None:
                connection_video_file = os.path.join(
                    config.validation_video_dir, f"{step}_connections.mp4"
                )
                connection_frames = make_connection_overlay_video(
                    val_state_grids,
                    np.asarray(edge_positions),
                    connection_video_file,
                    cell_stride=config.edge_visualization_stride,
                )
                connection_tensor = np.transpose(
                    connection_frames[np.newaxis, ...], (0, 1, 4, 2, 3)
                )
                tb_writer.add_video(
                    "val_connections_video",
                    vid_tensor=connection_tensor,
                    fps=30,
                    global_step=state.step,
                )

        if step % config.checkpoint_every == 0 and config.checkpoint_dir:
            # save checkpoint
            checkpoints.save_checkpoint(
                os.path.abspath(config.checkpoint_dir), state, step=state.step, keep=3
            )


def evaluate(config: NCAConfig, output_video_path: Optional[str] = None) -> None:
    """This function evaluates one Pokémon id for `config.total_eval_steps` steps.

    Args:
        config (NCAConfig): The config object.
        output_video_path (optional):
            Where to save the video path, (sometime like /abc/eval.mp4). Defaults to None.
            if none then the video is saved to `config.evaluation_video_file`.
    """
    pokemon_id = sample_evaluation_pokemon_id(
        config, jax.random.PRNGKey(config.eval_seed_random_seed)
    )
    pokemon_ids = jnp.array([pokemon_id], dtype=jnp.int32)
    evaluate_for_pokemon_id(
        config=config,
        output_video_path=output_video_path,
        pokemon_id=pokemon_id,
        pokemon_ids=pokemon_ids,
    )


def evaluate_for_pokemon_id(
    config: NCAConfig,
    output_video_path: Optional[str],
    pokemon_id: int = 0,
    pokemon_ids: Optional[Array] = None,
) -> None:
    """Evaluate one conditional id and save a cutout video.

    Args:
        config: Run-time config.
        output_video_path: Output file path for the rendered movie.
        pokemon_id: Which Pokémon id to condition on.
        pokemon_ids: Optional pre-broadcasted condition tensor.
    """

    if pokemon_ids is None:
        pokemon_ids = jnp.array([pokemon_id], dtype=jnp.int32)

    # create_state restores config.weights_dir only as initialization, then
    # restores config.checkpoint_dir on top of it. Do not restore weights_dir
    # again here: that would silently replace the trained inference checkpoint
    # with the older initialization checkpoint.
    state, _ = create_state(config)

    # Match training-time eval: use the non-jit cell update path so inference
    # videos are directly comparable to val videos logged during training.
    cell_update_fn = create_cell_update_fn(config, state.apply_fn, use_jit=False)

    dataset_generator = NCADataGenerator(
        pool_size=config.pool_size,
        batch_size=config.batch_size,
        dimensions=config.dimensions,
        model_output_len=config.model_output_len,
        seed_density=config.seed_density,
        seed_noise_density=config.seed_noise_density,
        seed_random_seed=config.seed_random_seed,
        seed_pattern=config.seed_pattern,
        seed_size=config.seed_size,
        pokemon_targets=config.pokemon_targets,
    )

    state_grid = np.array(dataset_generator.seed_state, copy=True)[np.newaxis, ...]
    if config.eval_seed_density > 0.0 or config.eval_seed_noise_density > 0.0:
        rng = np.random.default_rng(config.eval_seed_random_seed)
        # Keep the Poké Ball icon if configured and add random live cells for
        # additional entropy in evaluation.
        random_mask = rng.random(config.dimensions) < config.eval_seed_density
        if config.seed_pattern == "pokeball":
            random_mask &= np.asarray(state_grid[0, 3]) <= 0.0
        state_grid[0, 3:, random_mask] = 1.0

        if config.eval_seed_noise_density > 0.0:
            noise_mask = rng.random(config.dimensions) < config.eval_seed_noise_density
            if config.seed_pattern == "pokeball":
                noise_mask &= np.asarray(state_grid[0, 3]) <= 0.0
            state_grid[0, :3, noise_mask] = rng.random(
                (3, int(np.count_nonzero(noise_mask)))
            )
            state_grid[0, 3:, noise_mask] = 1.0
    key = jax.random.PRNGKey(0)
    # Keep evaluation RNG aligned with training-time eval, which uses an
    # unshifted seed at inference/eval call sites. Pokémon conditioning is
    # controlled explicitly through `pokemon_ids`.
    if config.inference_apply_cutout:
        state_grid_cache, _ = evaluate_damage_rollout(
            config=config,
            state=state,
            state_grid=state_grid,
            cell_update_fn=cell_update_fn,
            pokemon_ids=pokemon_ids,
            key=key,
        )
        state_grid_cache = jnp.squeeze(state_grid_cache)
    else:
        eval_result = evaluate_step(
            state=state,
            state_grid=state_grid,
            target=state_grid[:, :4],
            cell_update_fn=cell_update_fn,
            num_nca_steps=config.total_eval_steps,
            pokemon_ids=pokemon_ids,
            edge_count=(
                config.edge_count
                if config.nonlocal_connections
                and config.nonlocal_mode == "moving_edges"
                else 0
            ),
            edge_state_dim=config.edge_state_dim,
            return_edge_history=config.nonlocal_connections
            and config.nonlocal_mode == "moving_edges",
        )
        state_grid_array = eval_result[0] if isinstance(eval_result, tuple) else eval_result
        state_grid_cache = jnp.squeeze(state_grid_array)
    # Optional debug dump of rollout cache for local inspection.
    # Disabled by default to avoid unnecessary device-to-host transfers.
    if os.environ.get("NCA_SAVE_STATE_GRID_CACHE") == "1":
        np.save("/tmp/state_grid_cache.npy", np.asarray(state_grid_cache))

    state_grid_cache = jnp.clip(state_grid_cache, 0.0, 1.0)
    rgb = np.array(state_grid_cache)[:, :3]
    # Match training eval rendering (alpha thresholding then background composite).
    if rgb.ndim == 4:
        alpha = np.asarray(state_grid_cache)[:, 3:4] > 0.1
        rgb = alpha * np.asarray(state_grid_cache)[:, :3]
        # NCHW -> NHWC
        rgb = np.transpose(rgb, (0, 2, 3, 1))
    else:
        alpha = np.asarray(state_grid_cache)[..., 3:4] > 0.1
        rgb = alpha * np.asarray(state_grid_cache)[..., :3]

    if output_video_path is None:
        make_video(rgb, config.evaluation_video_file)
    else:
        make_video(rgb, output_video_path)


def evaluate_all_pokemon(config: NCAConfig, output_dir: str) -> None:
    """Run conditional inference and save one cutout MP4 per Pokémon id."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    pokemon_targets = tuple(config.pokemon_targets)
    if not pokemon_targets or pokemon_targets == (None,):
        pokemon_targets = (config.target_filename,)

    dataset_generator = NCADataGenerator(
        pool_size=config.pool_size,
        batch_size=config.batch_size,
        dimensions=config.dimensions,
        model_output_len=config.model_output_len,
        seed_density=config.seed_density,
        seed_random_seed=config.seed_random_seed,
        seed_pattern=config.seed_pattern,
        seed_size=config.seed_size,
        pokemon_targets=pokemon_targets,
    )

    total = len(pokemon_targets)
    if total == 0:
        raise ValueError("No Pokémon targets are configured for conditional inference")

    for pokemon_id in range(total):
        target_name = os.path.splitext(os.path.basename(pokemon_targets[pokemon_id]))[0]
        output_video_path = os.path.join(
            output_dir,
            f"pokemon_{pokemon_id:02d}_{target_name}_cutout.mp4",
        )
        evaluate_for_pokemon_id(
            config=config,
            output_video_path=output_video_path,
            pokemon_id=pokemon_id,
        )
