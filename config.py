import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    emb_dim: int = 512
    num_blocks: int = 8
    head_count: int = 8
    seq_length: int = 512
    ffn_multiple: int = 128


@dataclass
class TrainingConfig:
    token_count: int = 470_000_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "./checkpoints"
    