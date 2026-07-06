import torch
import torch.nn as nn

from configs.model import BetterGPTConfig as ModelConfig


class RoPESplitHalf(nn.Module):
    """Rotary Position Embedding — split-half variant (used by LLaMA).
    The head dimension is split into two halves; the rotation mixes the first
    half with the negated second half. compute sin/cos tables on the fly up to
    seq_len of each batch.
    """
    def __init__(self, head_dim: int, base: float):
        """
        Args:
            head_dim: Per-head dimension; must be even.
            base: Frequency base for the geometric sequence of inv-frequencies.
        """
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self._cos_cached = None
        self._sin_cached = None
        self._cached_len = 0

    def forward(self, x, seq_len):
        """Compute sin/cos tables for the given sequence length and device.
        Args:
            x: Input tensor of shape (batch_size, seq_len, head_dim).
            seq_len: Sequence length for which to compute sin/cos tables."""
        if (self._cos_cached is None or seq_len > self._cached_len
                or self._cos_cached.device != x.device):
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)     # (seq_len, head_dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)    # (seq_len, head_dim) — pairs with split-half
            self._cos_cached, self._sin_cached = emb.cos(), emb.sin()
            self._cached_len = seq_len
        return self._cos_cached[:seq_len].to(x.dtype), self._sin_cached[:seq_len].to(x.dtype)
    
