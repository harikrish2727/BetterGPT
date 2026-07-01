
import torch
import torch.nn as nn

from src.models.swiglu_feed_forward import SwiGLU_FFN
from src.models.attention import MHAttention
from src.models.layer_normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Single pre-norm transformer block.

    Applies: norm → attention → residual, then norm → FFN → residual.
    Pre-normalization (norm before the sub-layer rather than after) improves
    training stability for deep models.
    """

    def __init__(self, emb_dim, hid_dim, seq_length, rope, head_count, head_dim):
        """
        Args:
            emb_dim: Model embedding dimension.
            hid_dim: FFN hidden (expanded) dimension.
            seq_length: Maximum sequence length; passed to MHAttention for mask pre-computation.
            rope: Shared RoPE module applied inside attention.
            head_count: Number of attention heads.
            head_dim: Per-head dimension (emb_dim // head_count).
        """
        super().__init__()
        self.pre_attn_norm = RMSNorm(emb_dim)
        self.pre_ffn_norm = RMSNorm(emb_dim)
        self.attention = MHAttention(
            emb_dim=emb_dim,
            head_dim=head_dim,
            head_count=head_count,
            seq_length=seq_length,
            rope=rope,
        )
        self.ffn = SwiGLU_FFN(emb_dim=emb_dim, hid_dim=hid_dim)

    def forward(self, x, attention_mask):
        """Apply one transformer block with residual connections.

        Args:
            x: Input tensor of shape (B, T, emb_dim).
            attention_mask: Optional padding mask forwarded to MHAttention.

        Returns:
            Output tensor of shape (B, T, emb_dim).
        """
        x = x + self.attention(self.pre_attn_norm(x), attention_mask)
        x = x + self.ffn(self.pre_ffn_norm(x))
        return x
