import torch
from dataclasses import dataclass

from src.utils.paths import CHECKPOINT_DIR

checkpoint_dir = CHECKPOINT_DIR

#for smoke test
TARGET_TOTAL_TOKENS = 10_000_000_000  #smoke test
TOKEN_SWITCH_THRESHOLD = 9_000_000_000

@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for pretraining."""

    token_count: int = 10_000_000_000
    target_token_pretraining: int = TARGET_TOTAL_TOKENS
    token_switch_threshold: int = TOKEN_SWITCH_THRESHOLD
    batch_size: int = 8
    seq_length: int = 2048
    grad_accum_steps = 4
    learning_rate: float = 3e-4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = checkpoint_dir

    def __post_init__(self):
        if self.token_count <= 0:
            raise ValueError(f"token_count must be > 0, got {self.token_count}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
