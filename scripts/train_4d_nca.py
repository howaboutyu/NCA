"""Train the experimental Conv4D NCA against one RGBA target."""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training import checkpoints, train_state
from tensorboardX import SummaryWriter

from nca.dataset import NCADataGenerator
from nca.nca4d import (
    UpdateModel4D,
    project_xy_over_a_4d,
    rollout_4d,
    rollout_4d_frames,
    seed_volume_4d,
)
from nca.utils import make_gif


def foreground_weighted_mse(
    prediction: jax.Array, target: jax.Array, foreground_weight: float
) -> jax.Array:
    weights = 1.0 + (foreground_weight - 1.0) * target[:, 3:4]
    return jnp.mean(weights * jnp.square(prediction - target)) / jnp.mean(weights)


def rgba_to_rgb(face: np.ndarray) -> np.ndarray:
    face = np.clip(face, 0.0, 1.0)
    return np.transpose(face[:3] * face[3:4], (1, 2, 0))


def a_gallery_frame(volume: np.ndarray, z_index: int) -> np.ndarray:
    """Show exact A-axis slices side by side at one Z plane."""
    return np.concatenate(
        [rgba_to_rgb(volume[:, z_index, :, :, axis]) for axis in range(volume.shape[-1])],
        axis=1,
    )


def z_gallery_frame(volume: np.ndarray) -> np.ndarray:
    """Show A-composited XY projections for every Z slice."""
    return np.concatenate(
        [rgba_to_rgb(np.asarray(project_xy_over_a_4d(jnp.asarray(volume[None]), z))[0])
         for z in range(volume.shape[1])],
        axis=1,
    )


def evaluate_and_write_gifs(
    state: train_state.TrainState,
    model: UpdateModel4D,
    eval_seed: jax.Array,
    target: jax.Array,
    eval_key: jax.Array,
    config: dict,
    run_dir: str,
    writer: SummaryWriter,
    step: int,
) -> None:
    """Run a deterministic evaluation rollout and write 4D-aware GIFs."""
    frames = rollout_4d_frames(
        eval_key,
        eval_seed,
        model,
        state.params,
        config["nca_steps"],
        update_probability=1.0,
        state_clip=config.get("state_clip", 16.0),
    )
    frames_np = np.asarray(frames)
    z_index = config.get("projection_z_index", config["depth"] // 2)
    projection_frames = [
        rgba_to_rgb(np.asarray(project_xy_over_a_4d(frame, z_index)[0]))
        for frame in frames_np
    ]
    a_frames = [a_gallery_frame(frame[0], z_index) for frame in frames_np]
    z_frames = [z_gallery_frame(frame[0]) for frame in frames_np]
    eval_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    make_gif(projection_frames, os.path.join(eval_dir, f"{step:06d}_reconstruction.gif"), fps=8)
    make_gif(a_frames, os.path.join(eval_dir, f"{step:06d}_a_slices.gif"), fps=8)
    make_gif(z_frames, os.path.join(eval_dir, f"{step:06d}_z_slices.gif"), fps=8)
    for tag, gif_frames in (
        ("eval/reconstruction", projection_frames),
        ("eval/a_slices", a_frames),
        ("eval/z_slices", z_frames),
    ):
        video = np.transpose(np.asarray(gif_frames, dtype=np.float32)[None], (0, 1, 4, 2, 3))
        writer.add_video(tag, video, step, fps=8)
        writer.add_image(
            f"{tag}/first_frame",
            np.transpose(np.asarray(gif_frames[0]), (2, 0, 1)),
            step,
            dataformats="CHW",
        )

    final_projection = project_xy_over_a_4d(frames[-1], z_index)
    eval_loss = foreground_weighted_mse(
        final_projection, target, config.get("foreground_weight", 1.0)
    )
    alive = frames[-1][:, 3] > config.get("alive_threshold", 0.1)
    off_z = jnp.concatenate([alive[:, :z_index], alive[:, z_index + 1 :]], axis=1)
    writer.add_scalar("eval/reconstruction_loss", float(eval_loss), step)
    writer.add_scalar("eval/off_target_alive", float(jnp.mean(off_z)), step)
    writer.add_image("eval/reconstruction", np.asarray(final_projection[0]), step, dataformats="CHW")
    writer.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    height, width = config["dimensions"]
    run_dir = os.path.abspath(config["run_dir"])
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, "logs"))

    generator = NCADataGenerator(
        pool_size=1,
        batch_size=config["batch_size"],
        dimensions=(height, width),
        model_output_len=config["state_channels"],
    )
    target = generator.get_target(config["target_filename"])
    seed = seed_volume_4d(
        config["batch_size"],
        config["state_channels"],
        config["depth"],
        height,
        width,
        config["axis_size"],
        config.get("seed_depth_index", 0),
        config.get("seed_axis_index", 0),
        config.get("random_seed_density", 0.0),
        jax.random.PRNGKey(config.get("seed", 0)),
    )
    eval_seed = seed_volume_4d(
        config["batch_size"],
        config["state_channels"],
        config["depth"],
        height,
        width,
        config["axis_size"],
        config.get("eval_seed_depth_index", config.get("projection_z_index", config["depth"] // 2)),
        config.get("eval_seed_axis_index", (config.get("seed_axis_index", 0) + 1) % config["axis_size"]),
        0.0,
    )
    model = UpdateModel4D(
        state_channels=config["state_channels"],
        hidden_channels=config["hidden_channels"],
    )
    key = jax.random.PRNGKey(config.get("seed", 0))
    key, init_key = jax.random.split(key)
    params = model.init(
        init_key, jnp.transpose(seed, (0, 2, 3, 4, 5, 1))
    )
    schedule = optax.cosine_decay_schedule(
        config["learning_rate"], config["training_steps"]
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(optax.clip(0.5), optax.adam(schedule)),
    )

    @jax.jit
    def train_step(current_state, input_volume, train_target, step_key):
        def loss_fn(current_params):
            volume = rollout_4d(
                step_key,
                input_volume,
                model,
                current_params,
                config["nca_steps"],
                config["update_probability"],
                config.get("state_clip", 16.0),
            )
            projection = project_xy_over_a_4d(
                volume, config.get("projection_z_index", config["depth"] // 2)
            )
            reconstruction_loss = foreground_weighted_mse(
                projection, train_target, config.get("foreground_weight", 1.0)
            )
            alive = volume[:, 3] > config.get("alive_threshold", 0.1)
            z_index = config.get("projection_z_index", config["depth"] // 2)
            off_z = jnp.concatenate(
                [alive[:, :z_index], alive[:, z_index + 1 :]], axis=1
            )
            off_target_alive = jnp.mean(off_z.astype(jnp.float32))
            loss = reconstruction_loss + config.get("alive_volume_penalty", 0.0) * off_target_alive
            return loss, (projection, reconstruction_loss, off_target_alive)

        (loss, (projection, reconstruction_loss, off_target_alive)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(current_state.params)
        grads = jax.tree_util.tree_map(
            lambda gradient: gradient / (jnp.linalg.norm(gradient) + 1e-8), grads
        )
        return (
            current_state.apply_gradients(grads=grads),
            loss,
            projection,
            reconstruction_loss,
            off_target_alive,
        )

    last_projection = None
    for step in range(1, config["training_steps"] + 1):
        key, step_key = jax.random.split(key)
        state, loss, last_projection, reconstruction_loss, off_target_alive = train_step(
            state, seed, target, step_key
        )
        if step % config["log_every"] == 0 or step == 1:
            print(
                f"step={step} loss={float(loss):.6f} "
                f"reconstruction={float(reconstruction_loss):.6f} "
                f"off_target_alive={float(off_target_alive):.6f}",
                flush=True,
            )
            writer.add_scalar("loss", float(loss), step)
            writer.add_scalar("reconstruction_loss", float(reconstruction_loss), step)
            writer.add_scalar("off_target_alive", float(off_target_alive), step)
            writer.add_scalar("learning_rate", float(schedule(step)), step)
            writer.flush()
        if step % config["checkpoint_every"] == 0:
            checkpoints.save_checkpoint(
                checkpoint_dir, state, step=state.step, keep=3, overwrite=True
            )
        if step % config.get("eval_every", config["training_steps"] + 1) == 0:
            key, eval_key = jax.random.split(key)
            evaluate_and_write_gifs(
                state, model, eval_seed, target, eval_key, config, run_dir, writer, step
            )

    checkpoints.save_checkpoint(checkpoint_dir, state, step=state.step, keep=3, overwrite=True)
    if last_projection is not None:
        frame = rgba_to_rgb(np.asarray(last_projection[0]))
        make_gif([frame] * 12, os.path.join(run_dir, "projection.gif"), fps=6)
    writer.close()
    print(f"completed_steps={state.step} run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
