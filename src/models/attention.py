"""
class to create Multi-head self-attention with Rotary Position Embeddings (RoPE),
using pytorch SDPA class.

"""
import torch

from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from src.utils.logger import get_logger
from src.utils.rope_helper import apply_rotary_pos_emb

logger = get_logger(__name__)

class MHAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embeddings (RoPE).

    Uses PyTorch's scaled_dot_product_attention (flash-attention kernel when available).
    A lower-triangular causal mask is pre-computed at init time; an optional per-token
    attention_mask is AND-ed with it at forward time.
    """

    def __init__(
            self,
            emb_dim: int,
            head_count: int,
            head_dim: int
            ):
        """
        Args:
            emb_dim: Model embedding dimension.
            head_count: Number of attention heads.
            head_dim: Dimension per head (emb_dim // head_count).
        """
        super().__init__()
        self.head_count = head_count
        self.head_dim = head_dim
        self.emb_dim = emb_dim

        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x, sin, cos, attention_mask=None):
        """Apply multi-head attention with RoPE and an optional padding mask.

        Args:
            x: Input tensor of shape (B, T, emb_dim).
            sin: Sine values for RoPE.
            cos: Cosine values for RoPE.
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

        logger.debug(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        logger.debug(f"sin shape: {sin.shape}, cos shape: {cos.shape}")
        logger.debug(f"attention_mask shape: {
            attention_mask.shape if attention_mask is not None else 'None'}"
            )

        rotated_q, rotated_k = apply_rotary_pos_emb(q, k, cos, sin)

        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool),
            diagonal=1
        )  # Upper triangular mask for causal attention

        if attention_mask is not None:
            padding_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

            # Combine: mask positions that are padding OR in the future
            # causal_mask: [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) | (~padding_mask)
            #combined_mask is True where attention should be BLOCKED

            attn_bias = torch.zeros(batch, 1, seq_length, seq_length, device=x.device, dtype=q.dtype)
            attn_bias.masked_fill_(combined_mask, torch.finfo(q.dtype).min)

            y = scaled_dot_product_attention(
                rotated_q, rotated_k, v,
                attn_mask=attn_bias,
                is_causal=False
            )
        else:
            logger.debug("No attention_mask provided; using causal mask only.")
            y = scaled_dot_product_attention(rotated_q, rotated_k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch, seq_length, emb_dim)
        return self.out_proj(y)
