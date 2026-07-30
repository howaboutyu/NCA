"""Export a direct-synapse Flax checkpoint for the TensorFlow.js web runtime."""

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from nca.config import load_config
from nca.dataset import NCADataGenerator
from nca.trainer import create_cell_update_fn, create_state, nca_looper


PARAM_PATHS = {
    "perception.kernel": ("params", "perception", "kernel"),
    "perception.bias": ("params", "perception", "bias"),
    "edgeNet.0.kernel": ("params", "edge_net", "layers_0", "kernel"),
    "edgeNet.0.bias": ("params", "edge_net", "layers_0", "bias"),
    "edgeNet.2.kernel": ("params", "edge_net", "layers_2", "kernel"),
    "edgeNet.2.bias": ("params", "edge_net", "layers_2", "bias"),
    "embedding": ("params", "pokemon_embedding", "embedding"),
    "conv1.kernel": ("params", "conv_1", "kernel"),
    "conv1.bias": ("params", "conv_1", "bias"),
    "conv2.kernel": ("params", "conv_2", "kernel"),
    "conv2.bias": ("params", "conv_2", "bias"),
}


def nested_get(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mature-steps", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    restored = checkpoints.restore_checkpoint(
        os.path.abspath(args.checkpoint), target=None
    )
    params = restored["params"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    generator = NCADataGenerator(
        pool_size=1,
        batch_size=1,
        dimensions=config.dimensions,
        model_output_len=config.model_output_len,
        seed_density=config.seed_density,
        seed_noise_density=config.seed_noise_density,
        seed_random_seed=config.seed_random_seed,
        seed_pattern=config.seed_pattern,
        seed_size=config.seed_size,
        pokemon_targets=config.pokemon_targets,
    )
    seed_nhwc = np.transpose(generator.seed_state, (1, 2, 0))

    arrays = {
        name: np.asarray(nested_get(params, path), dtype="<f4")
        for name, path in PARAM_PATHS.items()
    }
    arrays["seed"] = np.asarray(seed_nhwc, dtype="<f4")

    # Pre-grow each target once so the browser opens on an immediately
    # damageable organism instead of spending its first seconds growing a seed.
    inference_config = load_config(args.config)
    inference_config.weights_dir = os.path.abspath(args.checkpoint)
    inference_config.checkpoint_dir = ""
    state, _ = create_state(inference_config)
    cell_update_fn = create_cell_update_fn(
        inference_config, state.apply_fn, use_jit=True
    )
    mature_states = []
    for pokemon_id in range(len(config.pokemon_targets)):
        _, sequence = nca_looper(
            key=jax.random.PRNGKey(100 + pokemon_id),
            params=state.params,
            state_grid=jnp.asarray(generator.seed_state[np.newaxis, ...]),
            num_nca_steps=args.mature_steps,
            cell_update_fn=cell_update_fn,
            pokemon_ids=jnp.asarray([pokemon_id], dtype=jnp.int32),
            edge_count=config.edge_count,
            edge_state_dim=config.edge_state_dim,
        )
        mature_states.append(
            np.transpose(np.asarray(sequence[-1, 0]), (1, 2, 0))
        )
    arrays["matureStates"] = np.asarray(mature_states, dtype="<f4")

    records = {}
    flat_arrays = []
    offset = 0
    for name, array in arrays.items():
        flat = np.ascontiguousarray(array).reshape(-1)
        records[name] = {
            "offset": offset,
            "length": int(flat.size),
            "shape": list(array.shape),
        }
        flat_arrays.append(flat)
        offset += int(flat.size)

    np.concatenate(flat_arrays).tofile(output / "weights.bin")
    checkpoint_name = Path(args.checkpoint).name
    checkpoint_step = int(checkpoint_name.rsplit("_", 1)[-1])
    manifest = {
        "checkpointStep": checkpoint_step,
        "dimensions": list(config.dimensions),
        "channels": config.model_output_len,
        "edgeCount": config.edge_count,
        "edgeStateDim": config.edge_state_dim,
        "edgeStepSize": config.edge_step_size,
        "edgeStateStepSize": config.edge_state_step_size,
        "edgeMessageScale": config.edge_message_scale,
        "targets": [Path(path).stem.title() for path in config.pokemon_targets],
        "weights": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Exported checkpoint {checkpoint_step}: "
        f"{(output / 'weights.bin').stat().st_size / 1_000_000:.2f} MB"
    )


if __name__ == "__main__":
    main()
