"""Build reproducible per-Pokemon and combined GIFs from 3D validation snapshots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import yaml
from PIL import Image


SNAPSHOT_PATTERN = re.compile(r"^(\d+)_rgb_voxel_sources\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--omit-combined",
        nargs="*",
        default=(),
        help="Pokemon names to omit only from the combined GIF.",
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def latest_snapshots(
    validation_dir: Path, pokemon_count: int, validation_every: int
) -> dict[int, tuple[int, Path]]:
    latest: dict[int, tuple[int, Path]] = {}
    for path in validation_dir.glob("*_rgb_voxel_sources.npz"):
        match = SNAPSHOT_PATTERN.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        pokemon_id = (step // validation_every) % pokemon_count
        if pokemon_id not in latest or step > latest[pokemon_id][0]:
            latest[pokemon_id] = (step, path)
    missing = sorted(set(range(pokemon_count)) - set(latest))
    if missing:
        raise RuntimeError(f"Missing validation snapshots for Pokemon IDs: {missing}")
    return latest


def render_frames(snapshot: Path, render_axis: str, render_index: int) -> list[np.ndarray]:
    with np.load(snapshot) as source:
        volumes = np.asarray(source["volumes"], dtype=np.float32)
        frame_indices = np.asarray(source["frame_indices"], dtype=np.int32)
    if volumes.ndim != 5 or volumes.shape[1] < 4:
        raise ValueError(f"Expected TCDHW RGBA volumes, got {volumes.shape}")
    if render_axis == "z":
        rgba = volumes[:, :4, render_index]
    elif render_axis == "y":
        rgba = volumes[:, :4, :, render_index]
    elif render_axis == "x":
        rgba = volumes[:, :4, :, :, render_index]
    else:
        raise ValueError("render_axis must be one of: x, y, z")
    rgba = np.clip(rgba, 0.0, 1.0)
    rgb = np.transpose(rgba[:, :3] * rgba[:, 3:4], (0, 2, 3, 1))
    frames = [(frame * 255.0).round().astype(np.uint8) for frame in rgb]
    if len(frames) != len(frame_indices):
        raise RuntimeError("Snapshot frame indices and volumes are misaligned")
    return frames


def labelled_frame(
    frame: np.ndarray, name: str, rollout_index: int, scale: int
) -> np.ndarray:
    image = cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    header = np.zeros((30, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{name} | t={rollout_index + 1}",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, image], axis=0)


def save_gif(frames: list[np.ndarray], output: Path, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot save an empty GIF")
    output.parent.mkdir(parents=True, exist_ok=True)
    palette_adaptive = getattr(Image, "Palette", Image).ADAPTIVE
    images = [
        Image.fromarray(frame).convert("P", palette=palette_adaptive, colors=128)
        for frame in frames
    ]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )


def combine_frames(frame_sets: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not frame_sets:
        raise ValueError("No Pokemon selected for the combined GIF")
    frame_count = min(map(len, frame_sets))
    if len(frame_sets) == 1:
        return frame_sets[0][:frame_count]
    rows = []
    for frame_index in range(frame_count):
        current = [frames[frame_index] for frames in frame_sets]
        if len(current) % 2:
            current.append(np.zeros_like(current[0]))
        row_images = [
            np.concatenate(current[offset : offset + 2], axis=1)
            for offset in range(0, len(current), 2)
        ]
        rows.append(np.concatenate(row_images, axis=0))
    return rows


def main() -> None:
    args = parse_args()
    if args.scale < 1 or args.fps < 1:
        raise ValueError("scale and fps must be positive")
    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    names = [Path(target).stem for target in config["pokemon_targets"]]
    snapshots = latest_snapshots(
        args.run_dir / "validation",
        len(names),
        int(config.get("validation_every", 500)),
    )
    omitted = {name.lower() for name in args.omit_combined}
    combined_inputs: list[list[np.ndarray]] = []
    manifest_lines = []
    for pokemon_id, name in enumerate(names):
        step, snapshot = snapshots[pokemon_id]
        raw_frames = render_frames(
            snapshot,
            config.get("render_axis", "z"),
            int(config.get("render_index", 0)),
        )
        frames = [
            labelled_frame(frame, name.title(), index, args.scale)
            for index, frame in enumerate(raw_frames)
        ]
        output = args.output_dir / f"pokemon_{pokemon_id:02d}_{name}_latest.gif"
        save_gif(frames, output, args.fps)
        manifest_lines.append(f"{pokemon_id}\t{name}\t{step}\t{snapshot}\t{output}")
        if name.lower() not in omitted:
            combined_inputs.append(frames)
    combined_output = args.output_dir / "conditional_pokemon_showcase_latest.gif"
    save_gif(combine_frames(combined_inputs), combined_output, args.fps)
    (args.output_dir / "sources.tsv").write_text(
        "pokemon_id\tname\tstep\tsnapshot\tgif\n" + "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    print(combined_output)


if __name__ == "__main__":
    main()
