"""Train a 3D NCA against a 2D RGBA target through depth compositing."""

import argparse
import os
import subprocess
import sys
from collections import deque

import cv2  # type: ignore
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import yaml
from flax.training import checkpoints, train_state

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from tensorboardX.proto.summary_pb2 import Summary

from nca.dataset import NCADataGenerator
from nca.nca3d import (
    UpdateModel3D,
    cell_update_3d,
    render_selected_plane_3d,
    rollout_3d,
    seed_volume_3d,
)
from nca.utils import encode_gif, make_gif, mse


def wave_slice_frame(volume: np.ndarray, slice_count: int = 4, scale: int = 4) -> np.ndarray:
    """Render enlarged numbered RGB/hidden-state slices along z."""
    volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(volume[:3], 0.0, 1.0)
    stride = max(1, volume.shape[1] // max(1, slice_count))
    selected_indices = list(range(0, volume.shape[1], stride))[:slice_count]
    visible = np.transpose(rgb[:, selected_indices], (1, 2, 3, 0))
    hidden = volume[4:] if volume.shape[0] > 4 else volume[3:4]
    energy = np.clip(np.tanh(np.mean(np.abs(hidden[:, selected_indices]), axis=0)), 0.0, 1.0)
    waves = np.stack([energy, 0.2 * energy, 1.0 - energy], axis=-1)
    enlarge = lambda image: cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    top = np.concatenate([enlarge(image) for image in visible], axis=1)
    bottom = np.concatenate([enlarge(image) for image in waves], axis=1)
    header = np.zeros((26, top.shape[1], 3), dtype=np.uint8)
    tile_width = visible.shape[2] * scale
    for output_index, slice_index in enumerate(selected_indices):
        cv2.putText(
            header,
            f"z={slice_index}",
            (output_index * tile_width + 3, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.clip(
        np.concatenate([header.astype(np.float32) / 255.0, top, bottom], axis=0),
        0.0,
        1.0,
    )


def rgba_face_to_rgb(face: np.ndarray) -> np.ndarray:
    """Render an NCA RGBA face exactly as the existing 2D visualizer does."""
    face = np.clip(face, 0.0, 1.0)
    return np.transpose(face[:3] * face[3:4], (1, 2, 0))


def foreground_weighted_mse(prediction: jax.Array, target: jax.Array, foreground_weight: float) -> jax.Array:
    """Balance object reconstruction against its transparent background."""
    weights = 1.0 + (foreground_weight - 1.0) * target[:, 3:4]
    return jnp.mean(weights * jnp.square(prediction - target)) / jnp.mean(weights)


def foreground_weighted_losses_np(
    prediction: np.ndarray, target: np.ndarray, foreground_weight: float
) -> np.ndarray:
    """Per-sample NumPy counterpart used for pool ranking."""
    weights = 1.0 + (foreground_weight - 1.0) * target[:, 3:4]
    numerator = np.mean(weights * np.square(prediction - target), axis=(1, 2, 3))
    denominator = np.mean(weights, axis=(1, 2, 3))
    return numerator / denominator


def validation_time_label(
    frame: np.ndarray, frame_index: int, annotation: str = ""
) -> np.ndarray:
    """Keep each validation GIF frame distinct and make rollout time explicit."""
    header = np.zeros((22, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"rollout t={frame_index + 1} {annotation}",
        (3, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.clip(
        np.concatenate([header.astype(np.float32) / 255.0, frame], axis=0), 0.0, 1.0
    )


def face_grid(target: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Create a labelled 2×2 target/reconstruction grid for TensorBoard."""
    faces = [("target", target)] + [
        (f"pool {index}", face) for index, face in enumerate(reconstructions[:3])
    ]
    tiles = []
    for label, face in faces:
        image = rgba_face_to_rgb(face)
        header = np.zeros((18, image.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, label, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(np.concatenate([header.astype(np.float32) / 255.0, image], axis=0))
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate(
        [np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:4], axis=1)], axis=0
    )


def pool_grid(
    pool_faces: np.ndarray, pool_indices: np.ndarray, citizen_ids: np.ndarray | None = None
) -> np.ndarray:
    """Render exactly 16 selected pool faces in a labelled 4×4 grid, no target."""
    if len(pool_faces) != 16:
        raise ValueError("pool_grid requires exactly 16 pool faces")
    tiles = []
    for offset, (pool_index, face) in enumerate(zip(pool_indices, pool_faces)):
        image = rgba_face_to_rgb(face)
        header = np.zeros((18, image.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"pool {int(pool_index)} id {int(citizen_ids[offset])}" if citizen_ids is not None else f"pool {int(pool_index)}",
            (2, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(np.concatenate([header.astype(np.float32) / 255.0, image], axis=0))
    rows = [np.concatenate(tiles[row : row + 4], axis=1) for row in range(0, 16, 4)]
    return np.concatenate(rows, axis=0)


def rgb_voxel_frame(
    volume: np.ndarray,
    frame_index: int,
    spatial_stride: int,
    alive_threshold: float,
    opacity: float,
) -> np.ndarray:
    """Render RGB state as translucent, edged voxels at live lattice cells only."""
    stride = max(1, spatial_stride)
    rgb = np.clip(np.nan_to_num(volume[:3], nan=0.0), 0.0, 1.0)
    alive = np.clip(np.nan_to_num(volume[3], nan=0.0), 0.0, 1.0)
    rgb = rgb[:, :, ::stride, ::stride]
    alive = alive[:, ::stride, ::stride]
    # Axes3D.voxels expects its volume in x, y, z order; NCA stores D, H, W.
    filled = (alive > alive_threshold).transpose(2, 1, 0)
    facecolors = rgb.transpose(3, 2, 1, 0)
    facecolors = np.concatenate(
        [facecolors, (opacity * alive.transpose(2, 1, 0))[..., None]], axis=-1
    )
    # Use one display unit per sampled lattice node, so every rendered voxel
    # is a cube rather than a 4×4×1 slab.
    x_edges = np.arange(filled.shape[0] + 1)
    y_edges = np.arange(filled.shape[1] + 1)
    z_edges = np.arange(filled.shape[2] + 1)
    x, y, z = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")

    figure = plt.figure(figsize=(6.4, 5.6), dpi=100, facecolor="#111827")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#111827")
    axis.voxels(
        x, y, z, filled, facecolors=facecolors,
        edgecolor=(0.92, 0.96, 1.0, 0.22), linewidth=0.12,
    )
    axis.set_xlim(0, filled.shape[0])
    axis.set_ylim(0, filled.shape[1])
    axis.set_zlim(0, filled.shape[2])
    axis.set_box_aspect(filled.shape)
    axis.view_init(elev=24, azim=35)
    axis.set_axis_off()
    figure.suptitle(
        f"live RGB voxels — rollout t={frame_index + 1}", color="white", y=0.95
    )
    figure.tight_layout()
    figure.canvas.draw()
    frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return frame.astype(np.float32) / 255.0


def damage_rendered_faces(
    volumes: np.ndarray,
    count: int,
    fraction: float,
    render_axis: str,
    render_index: int,
    strategies: tuple[str, ...],
    noise_probability: float,
    noise_scale: tuple[float, float],
    rng: np.random.Generator,
) -> None:
    """Apply 2D cutout strategies on the supervised rendered face."""
    if count <= 0:
        return
    _, _, depth, height, width = volumes.shape
    cut_height = max(1, int(round(height * fraction)))
    cut_width = max(1, int(round(width * fraction)))
    for volume in volumes[-count:]:
        if render_axis == "z":
            volume[:, render_index] = np.asarray(
                NCADataGenerator.random_cutout(
                    volume[:, render_index][None],
                    seed=int(rng.integers(0, 2**31 - 1)),
                    strategies=strategies,
                    square_height_factor_range=(fraction, fraction),
                    square_width_factor_range=(fraction, fraction),
                    left_width_factor_range=(0.25, 0.5),
                    noise_probability=noise_probability,
                    noise_scale=noise_scale,
                )[0]
            )
        elif render_axis == "y":
            x = rng.integers(0, width - cut_width + 1)
            volume[:, :, render_index, x : x + cut_width] = 0.0
        else:
            y = rng.integers(0, height - cut_height + 1)
            volume[:, :, y : y + cut_height, render_index] = 0.0


def write_gif_to_tensorboard(
    writer: SummaryWriter, tag: str, frames: list[np.ndarray], step: int, fps: int = 10
) -> None:
    """Write a real looping GIF Image summary, which TensorBoard renders natively."""
    encoded_gif = encode_gif(frames, fps)
    height, width = frames[0].shape[:2]
    summary = Summary(
        value=[
            Summary.Value(
                tag=tag,
                image=Summary.Image(
                    height=height,
                    width=width,
                    colorspace=3,
                    encoded_image_string=encoded_gif,
                ),
            )
        ]
    )
    writer._get_file_writer().add_summary(summary, step)


def launch_voxel_renderer(
    snapshot_path: str,
    output_path: str,
    log_dir: str,
    step: int,
    spatial_stride: int,
    alive_threshold: float,
    opacity: float,
) -> None:
    """Render expensive full-resolution voxel GIFs without blocking training."""
    renderer = os.path.join(os.path.dirname(__file__), "render_voxel_gif.py")
    pid_path = os.path.join(os.path.dirname(snapshot_path), "rgb_voxel_renderer.pid")
    try:
        with open(pid_path, "r", encoding="utf-8") as pid_file:
            existing_pid = int(pid_file.read().strip())
        os.kill(existing_pid, 0)
        # Preserve this snapshot for a later render, but never pile up CPU jobs.
        return
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    with open(f"{output_path}.render.log", "ab", buffering=0) as render_log:
        process = subprocess.Popen(
            [
                sys.executable, renderer,
                "--snapshot", snapshot_path,
                "--output", output_path,
                "--log-dir", log_dir,
                "--step", str(step),
                "--spatial-stride", str(spatial_stride),
                "--alive-threshold", str(alive_threshold),
                "--opacity", str(opacity),
                "--pid-file", pid_path,
            ],
            stdout=render_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    with open(pid_path, "w", encoding="utf-8") as pid_file:
        pid_file.write(str(process.pid))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    height, width = config["dimensions"]
    batch_size = config["batch_size"]
    state_channels = config["state_channels"]
    render_axis = config.get("render_axis", "z")
    render_index = config.get("render_index", 0)
    run_dir = os.path.abspath(config["run_dir"])
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(config.get("log_dir", os.path.join(run_dir, "logs")))
    # Separate event writers keep both large animated training summaries intact.
    log_dir = config.get("log_dir", os.path.join(run_dir, "logs"))
    training_face_writer = SummaryWriter(log_dir, filename_suffix=".training_face")
    training_grid_writer = SummaryWriter(log_dir, filename_suffix=".training_grid")
    voxel_writer = SummaryWriter(log_dir, filename_suffix=".rgb_voxels")

    generator = NCADataGenerator(
        pool_size=1,
        batch_size=batch_size,
        dimensions=(height, width),
        model_output_len=state_channels,
        pokemon_targets=tuple(config.get("pokemon_targets", ())),
    )
    targets_by_pokemon = generator.get_targets(config["target_filename"])
    pokemon_count = generator.pokemon_vocab_size
    target = targets_by_pokemon[0]
    writer.add_image(
        "target/rgba_face",
        rgba_face_to_rgb(np.asarray(target[0])).transpose(2, 0, 1),
        0,
    )
    writer.flush()
    model = UpdateModel3D(
        state_channels=state_channels,
        hidden_channels=config["hidden_channels"],
        pokemon_vocab_size=pokemon_count,
        pokemon_embedding_dim=config.get("pokemon_embedding_dim", 32),
        use_curl_divergence=config.get("use_curl_divergence", False),
    )
    seed_one = seed_volume_3d(
        1,
        state_channels,
        config["depth"],
        height,
        width,
        config.get("seed_depth_index", render_index),
        config.get("random_seed_density", 0.0),
        jax.random.PRNGKey(config.get("seed", 0)),
        config.get("seed_shape", "plane"),
        config.get("seed_radius", 1.0),
    )
    seed = jnp.repeat(seed_one, batch_size, axis=0)
    pool_size = config.get("pool_size", batch_size)
    state_pool = np.repeat(np.asarray(seed_one), pool_size, axis=0)
    # A citizen's identity is fixed for its entire lifetime in the pool.
    pool_pokemon_ids = np.arange(pool_size, dtype=np.int32) % pokemon_count
    displayed_pool_indices = np.arange(min(16, pool_size))
    if len(displayed_pool_indices) < 16:
        displayed_pool_indices = np.resize(displayed_pool_indices, 16)
    pool_grid_history: deque[np.ndarray] = deque(
        maxlen=config.get("pool_grid_history_frames", 64)
    )

    def record_pool_grid() -> None:
        pool_faces = np.asarray(
            render_selected_plane_3d(
                jnp.asarray(state_pool[displayed_pool_indices]),
                render_axis,
                render_index,
            )
        )
        pool_grid_history.append(
            pool_grid(
                pool_faces,
                displayed_pool_indices,
                pool_pokemon_ids[displayed_pool_indices],
            )
        )

    record_pool_grid()
    pool_rng = np.random.default_rng(config.get("seed", 0))
    key = jax.random.PRNGKey(config.get("seed", 0))
    key, init_key = jax.random.split(key)
    params = model.init(
        init_key,
        jnp.transpose(seed, (0, 2, 3, 4, 1)),
        pokemon_ids=jnp.asarray(pool_pokemon_ids[:batch_size]),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(
            optax.clip(0.5),
            optax.adam(
                optax.cosine_decay_schedule(
                    config["learning_rate"], config["training_steps"]
                )
            ),
        ),
    )
    learning_rate_schedule = optax.cosine_decay_schedule(
        config["learning_rate"], config["training_steps"]
    )
    resume_dir = config.get("resume_checkpoint_dir")
    if resume_dir:
        state = checkpoints.restore_checkpoint(os.path.abspath(resume_dir), state)
        print(f"resumed_from_step={state.step}", flush=True)

    @jax.jit
    def train_step(current_state, input_volume, train_target, pokemon_ids, step_key):
        def loss_fn(current_params):
            volume = rollout_3d(
                step_key,
                input_volume,
                model,
                current_params,
                config["nca_steps"],
                config["update_probability"],
                config.get("state_clip", 16.0),
                pokemon_ids=pokemon_ids,
            )
            projection = render_selected_plane_3d(
                volume, render_axis, render_index
            )
            reconstruction_loss = foreground_weighted_mse(
                projection, train_target, config.get("foreground_weight", 1.0)
            )
            alive = (volume[:, 3] > 0.1).astype(jnp.float32)
            if render_axis == "z":
                off_target_alive = jnp.concatenate(
                    [alive[:, :render_index], alive[:, render_index + 1 :]], axis=1
                )
            elif render_axis == "y":
                off_target_alive = jnp.concatenate(
                    [alive[:, :, :render_index], alive[:, :, render_index + 1 :]], axis=2
                )
            else:
                off_target_alive = jnp.concatenate(
                    [alive[:, :, :, :render_index], alive[:, :, :, render_index + 1 :]], axis=3
                )
            off_target_alive_fraction = jnp.mean(off_target_alive)
            loss = (
                reconstruction_loss
                + config.get("alive_volume_penalty", 0.0) * off_target_alive_fraction
            )
            return loss, (
                projection,
                volume,
                reconstruction_loss,
                off_target_alive_fraction,
            )

        (loss, (projection, volume, reconstruction_loss, alive_fraction)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            current_state.params
        )
        grads = jax.tree_util.tree_map(
            lambda gradient: gradient / (jnp.linalg.norm(gradient) + 1e-8), grads
        )
        return (
            current_state.apply_gradients(grads=grads),
            loss,
            projection,
            volume,
            reconstruction_loss,
            alive_fraction,
        )

    def write_validation(step: int) -> None:
        """Save the supervised face and enlarged numbered z slices as GIFs."""
        volume = seed_one
        pokemon_id = (step // max(1, config.get("validation_every", 1))) % pokemon_count
        validation_target = targets_by_pokemon[pokemon_id : pokemon_id + 1, 0]
        validation_pokemon_ids = jnp.asarray([pokemon_id], dtype=jnp.int32)
        selected_plane_frames, wave_frames, voxel_volumes, voxel_indices = [], [], [], []
        validation_key = jax.random.fold_in(key, step)
        cutout_rng = np.random.default_rng(int(step))
        validation_steps = config.get("validation_steps", config["nca_steps"])
        validation_cutout_steps = set(
            config.get(
                "validation_cutout_steps",
                [config.get("validation_cutout_step", -1)],
            )
        )
        voxel_frame_stride = max(1, int(config.get("rgb_voxel_frame_stride", 1)))
        voxel_frame_indices = set(range(0, validation_steps, voxel_frame_stride))
        for frame_index in range(validation_steps):
            validation_key, cell_key = jax.random.split(validation_key)
            volume = cell_update_3d(
                cell_key,
                volume,
                model,
                state.params,
                update_probability=config["update_probability"],
                state_clip=config.get("state_clip", 16.0),
                pokemon_ids=validation_pokemon_ids,
            )
            annotation = ""
            if frame_index + 1 in validation_cutout_steps:
                volume_np = np.asarray(volume).copy()
                damage_rendered_faces(
                    volume_np,
                    1,
                    config.get("validation_cutout_fraction", 0.2),
                    render_axis,
                    render_index,
                    tuple(config.get("cutout_strategies", ("circle",))),
                    0.0,
                    (0.0, 0.0),
                    cutout_rng,
                )
                volume = jnp.asarray(volume_np)
                annotation = "CUTOUT"
            selected_plane = render_selected_plane_3d(
                volume, render_axis, render_index
            )[0]
            selected_plane_frames.append(
                validation_time_label(
                    rgba_face_to_rgb(np.asarray(selected_plane)), frame_index, annotation
                )
            )
            volume_np = np.asarray(volume[0])
            wave_frames.append(
                validation_time_label(
                    wave_slice_frame(
                        volume_np,
                        config.get("depth_visualization_slices", 4),
                        config.get("slice_scale", 4),
                    ),
                    frame_index,
                    annotation,
                )
            )
            if frame_index in voxel_frame_indices:
                voxel_volumes.append(volume_np[:4].copy())
                voxel_indices.append(frame_index)

        validation_dir = os.path.join(run_dir, "validation")
        os.makedirs(validation_dir, exist_ok=True)
        make_gif(selected_plane_frames, os.path.join(validation_dir, f"{step}_selected_face.gif"), fps=10)
        make_gif(wave_frames, os.path.join(validation_dir, f"{step}_z_slices.gif"), fps=10)
        voxel_snapshot_path = os.path.join(validation_dir, f"{step}_rgb_voxel_sources.npz")
        np.savez_compressed(
            voxel_snapshot_path,
            volumes=np.stack(voxel_volumes),
            frame_indices=np.asarray(voxel_indices, dtype=np.int32),
        )
        launch_voxel_renderer(
            voxel_snapshot_path,
            os.path.join(validation_dir, f"{step}_rgb_voxels.gif"),
            log_dir,
            step,
            config.get("rgb_voxel_spatial_stride", 4),
            config.get("rgb_voxel_alive_threshold", 0.1),
            config.get("rgb_voxel_opacity", 0.45),
        )
        writer.add_scalar("validation/pokemon_id", pokemon_id, step)
        writer.add_image("target/rgba_face", rgba_face_to_rgb(np.asarray(validation_target[0])).transpose(2, 0, 1), step)
        writer.add_image(
            "validation/target_and_reconstruction",
            face_grid(np.asarray(validation_target[0]), np.asarray(selected_plane)[None]).transpose(2, 0, 1),
            step,
        )
        write_gif_to_tensorboard(
            writer, "validation/depth_slices", wave_frames, step
        )
        training_grid_frames = [
            validation_time_label(frame, frame_index)
            for frame_index, frame in enumerate(pool_grid_history)
        ]
        write_gif_to_tensorboard(
            training_face_writer, "training/selected_face", selected_plane_frames, step
        )
        training_face_writer.flush()
        write_gif_to_tensorboard(
            training_grid_writer,
            "training/grid_target_and_pool_reconstructions",
            training_grid_frames,
            step,
        )
        training_grid_writer.flush()
        writer.flush()

    if config.get("initial_validation", False):
        write_validation(state.step)

    last_projection = None
    for step in range(state.step, config["training_steps"]):
        key, step_key = jax.random.split(key)
        # Match the 2D branch: rank the sampled batch, reset its worst state,
        # damage its best states, then shuffle before updating the pool.
        # Rotate through target IDs and sample one distinct citizen per chosen
        # ID. This removes conditional-ID imbalance and duplicate pool slots.
        batch_pokemon_ids = (step + np.arange(batch_size, dtype=np.int32)) % pokemon_count
        pool_indices = np.asarray(
            [
                pool_rng.choice(np.flatnonzero(pool_pokemon_ids == pokemon_id))
                for pokemon_id in batch_pokemon_ids
            ],
            dtype=np.int32,
        )
        batch_volume_np = state_pool[pool_indices].copy()
        if not np.array_equal(batch_pokemon_ids, pool_pokemon_ids[pool_indices]):
            raise RuntimeError("balanced pool sampling selected a citizen with the wrong ID")
        batch_targets = np.asarray(targets_by_pokemon[batch_pokemon_ids, np.arange(batch_size)])
        current_faces = np.asarray(
            render_selected_plane_3d(
                jnp.asarray(batch_volume_np), render_axis, render_index
            )
        )
        loss_per_state = foreground_weighted_losses_np(
            current_faces, batch_targets, config.get("foreground_weight", 1.0)
        )
        ranked_indices = np.argsort(loss_per_state)[::-1]
        batch_volume_np = batch_volume_np[ranked_indices]
        pool_indices = pool_indices[ranked_indices]
        batch_pokemon_ids = batch_pokemon_ids[ranked_indices]
        batch_targets = batch_targets[ranked_indices]
        batch_volume_np[0] = np.asarray(seed_one[0])
        damage_rendered_faces(
            batch_volume_np,
            config.get("n_damage", 0),
            config.get("damage_fraction", 0.1),
            render_axis,
            render_index,
            tuple(config.get("cutout_strategies", ("circle",))),
            config.get("training_cutout_noise_probability", 0.0),
            tuple(config.get("training_cutout_noise_scale", (0.0, 0.0))),
            pool_rng,
        )
        shuffled_indices = pool_rng.permutation(batch_size)
        batch_volume = jnp.asarray(batch_volume_np[shuffled_indices])
        pool_indices = pool_indices[shuffled_indices]
        batch_pokemon_ids = batch_pokemon_ids[shuffled_indices]
        batch_targets = batch_targets[shuffled_indices]
        state, loss, last_projection, final_volume, reconstruction_loss, alive_fraction = train_step(
            state,
            batch_volume,
            jnp.asarray(batch_targets),
            jnp.asarray(batch_pokemon_ids),
            step_key,
        )
        state_pool[pool_indices] = np.asarray(final_volume)
        if step % config.get("pool_grid_history_stride", 8) == 0:
            record_pool_grid()
        if step % config["log_every"] == 0:
            print(f"step={step} loss={float(loss):.6f}", flush=True)
            writer.add_scalar("loss", float(loss), step)
            writer.add_scalar("reconstruction_loss", float(reconstruction_loss), step)
            writer.add_scalar("alive_fraction", float(alive_fraction), step)
            writer.add_scalar(
                "alive_volume_penalty",
                float(config.get("alive_volume_penalty", 0.0) * alive_fraction),
                step,
            )
            writer.add_scalar("learning_rate", learning_rate_schedule(state.step), step)
            # Scalars must remain visible while the following validation GIF
            # render is still running, which can take several minutes.
            writer.flush()
        if step > 0 and step % config.get("validation_every", config["training_steps"]) == 0:
            write_validation(step)
            checkpoints.save_checkpoint(
                os.path.join(run_dir, "checkpoints"), state, step=state.step, keep=3
            )

    checkpoints.save_checkpoint(
        os.path.join(run_dir, "checkpoints"),
        state,
        step=state.step,
        keep=1,
        overwrite=True,
    )
    if last_projection is not None:
        frame = rgba_face_to_rgb(np.asarray(last_projection[0]))
        make_gif([frame] * 12, os.path.join(run_dir, "selected_face.gif"), fps=6)

    write_validation(state.step)
    writer.close()
    training_face_writer.close()
    training_grid_writer.close()
    voxel_writer.close()


if __name__ == "__main__":
    main()
