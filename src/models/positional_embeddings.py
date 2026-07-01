
import torch
import torch.nn as nn


class RoPE_Interleave(nn.Module):
    """Rotary Position Embedding — interleaved variant.

    Sin/cos values are expanded with repeat_interleave so each pair of adjacent
    elements (even, odd) in the head dimension shares a frequency. Pre-computes
    sin/cos tables up to max_seq_len at init time.
    """

    def __init__(self, max_seq_len, head_dim, base=10000):
        """
        Args:
            max_seq_len: Maximum sequence length to pre-compute tables for.
            head_dim: Per-head dimension; must be even.
            base: Frequency base for the geometric sequence of inv-frequencies.
        """
        super().__init__()

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        pos = torch.arange(max_seq_len).float()
        angles = torch.outer(pos, inv_freq)

        sin = angles.sin().repeat_interleave(2, dim=-1)
        cos = angles.cos().repeat_interleave(2, dim=-1)

        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def rotate_half(self, x):
        """Swap adjacent pairs with a sign flip: [-x1, x0, -x3, x2, ...]."""
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def forward(self, x):
        """Apply RoPE rotation to query or key tensor x of shape (B, H, T, head_dim)."""
        T = x.size(-2)
        sin = self.sin[:T]
        cos = self.cos[:T]
        return x * cos + self.rotate_half(x) * sin


class RoPESplitHalf(nn.Module):
    """Rotary Position Embedding — split-half variant (used by LLaMA).

    The head dimension is split into two halves; the rotation mixes the first
    half with the negated second half. Pre-computes sin/cos tables up to
    max_seq_len at init time. Buffers are non-persistent (not saved to checkpoints).
    """

    def __init__(self, head_dim, max_seq_len, base=10000):
        """
        Args:
            head_dim: Per-head dimension; must be even.
            max_seq_len: Maximum sequence length to pre-compute tables for.
            base: Frequency base for the geometric sequence of inv-frequencies.
        """
        super().__init__()

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        pos = torch.arange(max_seq_len).float()
        angles = torch.outer(pos, inv_freq)

        sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def rotate_half(self, x):
        """Return [-x2 | x1] where x is split into equal halves x1 and x2."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        """Apply RoPE rotation to query or key tensor x of shape (B, H, T, head_dim)."""
        T = x.size(-2)
        sin = self.sin[:T]
        cos = self.cos[:T]
        return x * cos + self.rotate_half(x) * sin
