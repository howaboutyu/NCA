from dataclasses import dataclass
import yaml  # type: ignore


@dataclass
class NCAConfig:
    dimensions: tuple = (20,  20)
    model_output_len: int = 32 
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
    use_non_local_perceive: bool = False

    # evaluation parameters
    total_eval_steps: int = 300
    evaluation_video_file: str = "/tmp/evaluation_video.mp4"


def load_config(config_file: str) -> NCAConfig:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return NCAConfig(**config)


def write_config(config: NCAConfig, config_file: str) -> None:
    with open(config_file, "w") as f:
        yaml.dump(config.__dict__, f)
