"""Background renderer for validation RGB voxel GIFs."""

import argparse
import os

import numpy as np
from tensorboardX import SummaryWriter

from nca.utils import make_gif
from scripts.train_3d_nca import rgb_voxel_frame, write_gif_to_tensorboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--spatial-stride", required=True, type=int)
    parser.add_argument("--alive-threshold", required=True, type=float)
    parser.add_argument("--opacity", required=True, type=float)
    parser.add_argument("--pid-file", required=True)
    args = parser.parse_args()

    try:
        snapshot = np.load(args.snapshot)
        frames = [
            rgb_voxel_frame(
                volume,
                int(frame_index),
                args.spatial_stride,
                args.alive_threshold,
                args.opacity,
            )
            for volume, frame_index in zip(snapshot["volumes"], snapshot["frame_indices"])
        ]
        make_gif(frames, args.output, fps=4)
        writer = SummaryWriter(args.log_dir, filename_suffix=f".rgb_voxels_{args.step}")
        write_gif_to_tensorboard(writer, "validation/rgb_voxels", frames, args.step, fps=4)
        writer.flush()
        writer.close()
    finally:
        try:
            with open(args.pid_file, "r", encoding="utf-8") as pid_file:
                owns_lock = pid_file.read().strip() == str(os.getpid())
            if owns_lock:
                os.remove(args.pid_file)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
