from dataclasses import dataclass
import yaml  # type: ignore


@dataclass
class NCAConfig:
    dimensions: tuple = (40, 40)
    model_output_len: int = 16
    batch_size: int = 16
    num_steps: int = 100000
    eval_every: int = 30
    learning_rate: float = 1e-4
    pool_size: int = 1000
    target_filename: str = "emoji_imgs/smile.png"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 30
    validation_video_dir: str = "validation_videos"
    log_dir: str = "logs"
    log_every: int = 10


def load_config(config_file: str) -> NCAConfig:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return NCAConfig(**config)
