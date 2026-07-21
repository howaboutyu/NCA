import os
from dataclasses import dataclass
import yaml  # type: ignore


@dataclass
class NCAConfig:
    dimensions: tuple = (56, 56)
    model_output_len: int = 16
    # ``sobel`` is the original implementation. ``sobel_fused`` computes the
    # same features with one convolution, ``sobel_second`` adds second
    # derivatives, while ``learned`` trains a 3x3 perception convolution.
    perception_method: str = "sobel"
    nonlocal_connections: bool = False
    nonlocal_mode: str = "global"
    nonlocal_token_grid: int = 8
    nonlocal_attention_dim: int = 32
    # Optional differentiable moving-edge communication.
    edge_count: int = 4
    edge_state_dim: int = 16
    edge_momentum: float = 0.9
    edge_step_size: float = 0.05
    edge_state_step_size: float = 0.05
    edge_message_scale: float = 0.25
    edge_visualization_stride: int = 4
    state_clip: float = 16.0
    pokemon_targets: tuple = ()
    pokemon_embedding_dim: int = 32
    # Initial living-cell pattern. ``single`` preserves the original seed;
    # ``random`` uses ``seed_density``; ``pokeball`` uses a compact RGBA icon.
    seed_pattern: str = "single"
    seed_size: int = 11
    # Fraction of grid cells initialized as living cells. Zero preserves the
    # original single-cell center seed.
    seed_density: float = 0.0
    # For a Poké Ball seed, optionally add random live cells around the icon.
    seed_noise_density: float = 0.0
    seed_random_seed: int = 0
    batch_size: int = 16
    total_training_steps: int = 100000
    eval_every: int = 500
    learning_rate: float = 2e-4
    pool_size: int = 1000
    stochastic_update_prob: float = 0.5
    target_filename: str = "emoji_imgs/skier.png"
    weights_dir: str = "checkpoints"  # where to load ckpt
    checkpoint_dir: str = "checkpoints"  # where to save ckpt
    checkpoint_every: int = 500
    validation_video_dir: str = "validation_videos"
    log_dir: str = "logs"
    log_every: int = 500
    num_nca_steps: int = 64  # number of steps to run NCA for
    n_damage: int = 3  # number of states in a batch to damage
    # Kept for compatibility with older YAML files.  New configurations
    # should use ``n_damage`` to control damage augmentation.
    damage: bool = False

    # evaluation parameters
    total_eval_steps: int = 300
    evaluation_video_file: str = "/tmp/evaluation_video.mp4"


def load_config(config_file: str) -> NCAConfig:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cfg = NCAConfig(**config)
    config_dir = os.path.dirname(os.path.abspath(config_file))
    config_root = os.path.dirname(config_dir)

    def _resolve_path(value: str | None) -> str:
        if not value:
            return value or ""
        if os.path.isabs(value):
            return value
        absolute_candidate = os.path.abspath(value)
        if os.path.exists(absolute_candidate):
            return absolute_candidate
        config_dir_candidate = os.path.abspath(os.path.join(config_dir, value))
        if os.path.exists(config_dir_candidate):
            return config_dir_candidate
        root_dir_candidate = os.path.abspath(os.path.join(config_root, value))
        if os.path.exists(root_dir_candidate):
            return root_dir_candidate
        return config_dir_candidate

    cfg.target_filename = _resolve_path(cfg.target_filename)
    cfg.weights_dir = _resolve_path(cfg.weights_dir)
    cfg.checkpoint_dir = _resolve_path(cfg.checkpoint_dir)
    cfg.validation_video_dir = _resolve_path(cfg.validation_video_dir)
    cfg.log_dir = _resolve_path(cfg.log_dir)
    cfg.evaluation_video_file = _resolve_path(cfg.evaluation_video_file)

    if cfg.pokemon_targets:
        cfg.pokemon_targets = tuple(_resolve_path(path) for path in cfg.pokemon_targets)

    return cfg


def write_config(config: NCAConfig, config_file: str) -> None:
    with open(config_file, "w") as f:
        yaml.dump(config.__dict__, f)
