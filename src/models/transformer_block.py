import torch.nn as nn

from src.models.swiglu_feed_forward import SwiGLU_FFN
from src.models.attention import MHAttention
from src.models.layer_normalization import RMSNorm
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransformerBlock(nn.Module):
    """Single pre-norm transformer block.

    Applies: norm → attention → residual, then norm → FFN → residual.
    Pre-normalization (norm before the sub-layer rather than after) improves
    training stability for deep models.
    """

    def __init__(
        self, emb_dim: int, hid_dim: int, head_count: int, head_dim: int, eps: float
    ):
        """
        Args:
            emb_dim: Model embedding dimension.
            hid_dim: FFN hidden (expanded) dimension.
            head_count: Number of attention heads.
            head_dim: Per-head dimension (emb_dim // head_count).
        """
        super().__init__()
        self.pre_attn_norm = RMSNorm(emb_dim, eps)
        self.pre_ffn_norm = RMSNorm(emb_dim, eps)
        self.attention = MHAttention(
            emb_dim=emb_dim, head_dim=head_dim, head_count=head_count
        )
        self.ffn = SwiGLU_FFN(emb_dim=emb_dim, hid_dim=hid_dim)

    def forward(self, x, sin, cos, attention_mask):
        """Apply one transformer block with residual connections.

        Args:
            x: Input tensor of shape (B, T, emb_dim).
            sin: Sine values for rotary position embedding.
            cos: Cosine values for rotary position embedding.
            attention_mask: Optional padding mask forwarded to MHAttention.

        Returns:
            Output tensor of shape (B, T, emb_dim).
        """
        logger.debug(f"Input shape: {x.shape}")
        x = x + self.attention(self.pre_attn_norm(x), sin, cos, attention_mask)
        x = x + self.ffn(self.pre_ffn_norm(x))
        logger.debug(f"Output shape: {x.shape}")
        return x
