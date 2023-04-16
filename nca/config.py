from dataclasses import dataclass


@dataclass
class NCAConfig:
    dimensions: tuple = (40, 40)
    model_output_len: int = 16
    batch_size: int = 16
    num_epochs: int = 100000
    steps_per_epoch: int = 100
    learning_rate: float = 1e-4
