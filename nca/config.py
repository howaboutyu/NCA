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
    # Deprecated compatibility field. Moving edges now predict direct (dx, dy)
    # displacements and do not carry velocity between NCA steps.
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

    # Evaluation-time overrides for initialization.
    # Keep these at 0/empty to preserve current evaluation behavior.
    eval_seed_density: float = 0.0
    eval_seed_noise_density: float = 0.0
    eval_seed_random_seed: int = 0
    # Sample a different configured conditional target for each training-time
    # validation rollout. Explicit ``evaluate_for_pokemon_id`` calls are never
    # overridden.
    eval_random_pokemon_id: bool = True
    # Inference rollout behavior.
    # False => match training eval (single contiguous rollout, no cutout perturbations).
    # True  => apply randomized cutouts during inference rollout.
    inference_apply_cutout: bool = False
    # Legacy fixed-size inference settings (kept for backward compatibility with
    # older YAML files). New configs should use the *_range variants below.
    inference_cutout_height_factor: float = 0.2
    inference_cutout_width_factor: float = 0.2
    inference_cutout_strategies: tuple = ("left_side",)
    inference_cutout_square_height_factor_range: tuple = (0.05, 0.2)
    inference_cutout_square_width_factor_range: tuple = (0.05, 0.2)
    inference_cutout_left_width_factor_range: tuple = (0.25, 0.5)
    inference_cutout_noise_probability: float = 0.0
    inference_cutout_noise_scale: tuple = (0.0, 0.2)
    # Schedule qualitative regeneration damage during evaluation rollouts.
    # The first cut is applied before ``start_step`` is updated, then repeats
    # every ``interval_steps``. Values <= 0 disable the corresponding schedule.
    inference_cutout_start_step: int = 64
    inference_cutout_interval_steps: int = 64
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
    # Randomized damage augmentation options.
    train_cutout_strategies: tuple = ("circle", "left_side", "square")
    train_cutout_square_height_factor_range: tuple = (0.05, 0.2)
    train_cutout_square_width_factor_range: tuple = (0.05, 0.2)
    train_cutout_left_width_factor_range: tuple = (0.2, 0.5)
    train_cutout_noise_probability: float = 0.0
    train_cutout_noise_scale: tuple = (0.0, 0.2)
    # Kept for compatibility with older YAML files.  New configurations
    # should use ``n_damage`` to control damage augmentation.
    # If this key is present in YAML, ``False`` disables damage augmentation
    # by forcing ``n_damage`` to 0.
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

    if "damage" in config:
        if config["damage"]:
            if cfg.n_damage <= 0:
                cfg.n_damage = 1
        else:
            cfg.n_damage = 0

    if cfg.n_damage < 0:
        cfg.n_damage = 0

    if "inference_cutout_left_width_factor_range" not in config:
        cfg.inference_cutout_left_width_factor_range = (
            cfg.inference_cutout_width_factor,
            cfg.inference_cutout_width_factor,
        )
    if "inference_cutout_square_height_factor_range" not in config:
        cfg.inference_cutout_square_height_factor_range = (
            cfg.inference_cutout_height_factor,
            cfg.inference_cutout_height_factor,
        )
    if "inference_cutout_square_width_factor_range" not in config:
        cfg.inference_cutout_square_width_factor_range = (
            cfg.inference_cutout_width_factor,
            cfg.inference_cutout_width_factor,
        )
    if "train_cutout_strategies" not in config:
        cfg.train_cutout_strategies = ("circle", "left_side", "square")
    if "train_cutout_square_height_factor_range" not in config:
        cfg.train_cutout_square_height_factor_range = (0.05, 0.2)
    if "train_cutout_square_width_factor_range" not in config:
        cfg.train_cutout_square_width_factor_range = (0.05, 0.2)
    if "train_cutout_left_width_factor_range" not in config:
        cfg.train_cutout_left_width_factor_range = (0.2, 0.5)
    if "train_cutout_noise_scale" not in config:
        cfg.train_cutout_noise_scale = (0.0, 0.2)
    if "inference_cutout_strategies" not in config:
        cfg.inference_cutout_strategies = ("left_side",)
    if "inference_cutout_noise_scale" not in config:
        cfg.inference_cutout_noise_scale = (0.0, 0.2)

    return cfg


def write_config(config: NCAConfig, config_file: str) -> None:
    with open(config_file, "w") as f:
        yaml.dump(config.__dict__, f)
