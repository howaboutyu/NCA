from dataclasses import dataclass
import yaml  # type: ignore


@dataclass
class NCAConfig:
    dimensions: tuple = (56, 56)
    model_output_len: int = 16
    batch_size: int = 16
    total_training_steps: int = 100000
    eval_every: int = 500
    learning_rate: float = 2e-4
    pool_size: int = 1000
    target_filename: str = "emoji_imgs/skier.png"
    weights_dir: str = "checkpoints"  # where to load ckpt
    checkpoint_dir: str = "checkpoints"  # where to save ckpt
    checkpoint_every: int = 500
    validation_video_dir: str = "validation_videos"
    log_dir: str = "logs"
    log_every: int = 500
    num_nca_steps: int = 64  # number of steps to run NCA for
    n_damage: int = 3  # number of states in a batch to damage

    # evaluation parameters
    total_eval_steps: int = 300
    evaluation_video_file: str = "/tmp/evaluation_video.mp4"


def load_config(config_file: str) -> NCAConfig:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return NCAConfig(**config)
