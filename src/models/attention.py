
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


class MHAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embeddings (RoPE).

    Uses PyTorch's scaled_dot_product_attention (flash-attention kernel when available).
    A lower-triangular causal mask is pre-computed at init time; an optional per-token
    attention_mask is AND-ed with it at forward time.
    """

    def __init__(self, emb_dim, head_count, head_dim, seq_length, rope):
        """
        Args:
            emb_dim: Model embedding dimension.
            head_count: Number of attention heads.
            head_dim: Dimension per head (emb_dim // head_count).
            seq_length: Maximum sequence length for pre-computing the causal mask.
            rope: RoPE module applied to queries and keys before attention.
        """
        super().__init__()

        self.head_count = head_count
        self.head_dim = head_dim
        self.rope = rope

        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_length, seq_length)),
            persistent=False,
        )

    def forward(self, x, attention_mask):
        """Apply multi-head attention with RoPE and an optional padding mask.

        Args:
            x: Input tensor of shape (B, T, emb_dim).
            attention_mask: Optional bool tensor (B, T); True = attend, False = mask out.

        Returns:
            Output tensor of shape (B, T, emb_dim).
        """
        batch, seq_length, emb_dim = x.shape
        qkv = self.qkv_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_length, self.head_count, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_length, self.head_count, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_length, self.head_count, self.head_dim).transpose(1, 2)

        rotated_q = self.rope(q)
        rotated_k = self.rope(k)

        mask = self.mask[:seq_length, :seq_length].bool()

        if attention_mask is not None:
            attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(1)
            combined_mask = attention_mask & mask
            attn_bias = torch.zeros_like(combined_mask, dtype=q.dtype)
            attn_bias.masked_fill_(~combined_mask, torch.finfo(q.dtype).min)
            y = scaled_dot_product_attention(rotated_q, rotated_k, v, attn_mask=attn_bias, is_causal=False)
        else:
            y = scaled_dot_product_attention(rotated_q, rotated_k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch, seq_length, emb_dim)
        y = self.out_proj(y)
        return y
