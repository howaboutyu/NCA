"""Compare NCA perception implementations on the same short training run.

Example:
    python scripts/benchmark_perception.py --steps 100 --nca-steps 32
"""

import argparse
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp

# Allow this file to be run directly from the repository root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nca.config import NCAConfig
from nca.dataset import NCADataGenerator
from nca.trainer import create_cell_update_fn, create_state, evaluate_step, train_step


def benchmark(method: str, steps: int, nca_steps: int, dimensions: tuple[int, int]):
    global_context = method == "global_context"
    config = NCAConfig(
        dimensions=dimensions,
        model_output_len=16,
        batch_size=8,
        total_training_steps=steps,
        num_nca_steps=nca_steps,
        # Measure accuracy at the same propagation horizon used for training.
        total_eval_steps=nca_steps,
        target_filename="emoji_imgs/smile.png",
        weights_dir="",
        checkpoint_dir="",
        perception_method="sobel" if global_context else method,
        nonlocal_connections=global_context,
    )
    state, _ = create_state(config)
    update_fn = create_cell_update_fn(config, state.apply_fn, use_jit=False)
    train_fn = jax.jit(
        lambda key, train_state, grid, target: train_step(
            key, train_state, grid, target, update_fn, num_nca_steps=nca_steps
        )
    )

    generator = NCADataGenerator(
        pool_size=config.batch_size,
        batch_size=config.batch_size,
        dimensions=dimensions,
        model_output_len=config.model_output_len,
    )
    target = generator.get_target(config.target_filename)
    state_grid = jnp.asarray(generator.pool)
    key = jax.random.PRNGKey(0)

    # The first call includes XLA compilation and is reported separately.
    start = time.perf_counter()
    state, loss, _ = train_fn(key, state, state_grid, target)
    jax.block_until_ready(loss)
    compile_and_first_step_s = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(steps - 1):
        key, step_key = jax.random.split(key)
        state, loss, _ = train_fn(step_key, state, state_grid, target)
        jax.block_until_ready(loss)
    steady_state_s = time.perf_counter() - start

    _, eval_loss = evaluate_step(
        state,
        state_grid[:1],
        target[:1],
        update_fn,
        num_nca_steps=config.total_eval_steps,
    )
    jax.block_until_ready(eval_loss)

    return {
        "method": method,
        "steps": steps,
        "nca_steps": nca_steps,
        "dimensions": list(dimensions),
        "compile_and_first_step_s": compile_and_first_step_s,
        "steady_state_total_s": steady_state_s,
        "steady_state_step_s": steady_state_s / max(steps - 1, 1),
        "train_loss": float(loss),
        # evaluate_step already reduces the final propagated RGBA prediction
        # against the target, avoiding accidental broadcasting over time.
        "eval_mse": float(eval_loss),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--nca-steps", type=int, default=32)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "sobel",
            "sobel_fused",
            "sobel_second",
            "sobel_multiscale",
            "global_context",
            "learned",
        ],
    )
    args = parser.parse_args()

    print(
        json.dumps({"jax": jax.__version__, "devices": [str(d) for d in jax.devices()]})
    )
    results = [
        benchmark(method, args.steps, args.nca_steps, (args.height, args.width))
        for method in args.methods
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
