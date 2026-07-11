from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TokenizerConfig:
    special_tokens: List[str]
    vocab_size: int = 32768
    batch_size: int = 1000
    min_frequency: int = 2
    text_field: str = "text"
    model_max_length: int = 2048
    unk_token: Optional[str] = "<|unk|>"
    pad_token: Optional[str] = "<|pad|>"
    bos_token: Optional[str] = "<|bos|>"
    eos_token: Optional[str] = "<|eos|>"
    additional_special_tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.special_tokens:
            raise ValueError("special_tokens must be non-empty")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens contains duplicates")
        for t in self.special_tokens:
            if not t or t.strip() != t:
                raise ValueError(f"invalid special token: {t!r}")
        if self.vocab_size <= len(self.special_tokens):
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must exceed "
                f"len(special_tokens) ({len(self.special_tokens)}) to allow merges"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        # Required HF special tokens must be in special_tokens
        for name in ("unk_token", "pad_token", "bos_token", "eos_token"):
            tok = getattr(self, name)
            if tok is not None and tok not in self.special_tokens:
                raise ValueError(
                    f"{name}={tok!r} not in special_tokens; "
                    f"either add it or set {name}=None"
                )
