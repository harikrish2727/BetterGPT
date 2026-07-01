from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    emb_dim: int = 512
    num_blocks: int = 8
    head_count: int = 8
    seq_length: int = 512
    ffn_multiple: int = 128

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.emb_dim <= 0:
            raise ValueError(f"emb_dim must be > 0, got {self.emb_dim}")
        if self.head_count <= 0:
            raise ValueError(f"head_count must be > 0, got {self.head_count}")
        if self.emb_dim % self.head_count != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by head_count ({self.head_count})"
            )
        head_dim = self.emb_dim // self.head_count
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be even for RoPE; adjust emb_dim or head_count"
            )
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {self.num_blocks}")
        if self.seq_length <= 0:
            raise ValueError(f"seq_length must be > 0, got {self.seq_length}")
        if self.ffn_multiple <= 0:
            raise ValueError(f"ffn_multiple must be > 0, got {self.ffn_multiple}")
