from dataclasses import dataclass
import yaml  # type: ignore


@dataclass
class NCAConfig:
    dimensions: tuple = (56, 56)
    model_output_len: int = 16
    batch_size: int = 16
    num_steps: int = 100000
    eval_every: int = 500
    learning_rate: float = 2e-4
    pool_size: int = 1000
    target_filename: str = "emoji_imgs/skier.png"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 500
    validation_video_dir: str = "validation_videos"
    log_dir: str = "logs"
    log_every: int = 500


def load_config(config_file: str) -> NCAConfig:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return NCAConfig(**config)
