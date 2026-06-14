from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    emb_dim: int = 64
    num_blocks: int = 8
    head_count: int = 8
    seq_length: int = 64
    ffn_multiple: int = 128