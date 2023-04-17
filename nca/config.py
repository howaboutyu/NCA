from dataclasses import dataclass


@dataclass
class NCAConfig:
    dimensions: tuple = (40, 40)
    model_output_len: int = 16
    batch_size: int = 16
    num_steps: int = 100000
    learning_rate: float = 1e-4
    pool_size: int = 100
    target_filename: str = "emoji_imgs/skier.png"
