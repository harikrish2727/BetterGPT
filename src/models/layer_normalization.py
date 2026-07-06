import torch
import torch.nn as nn

from configs.model import BetterGPTConfig as ModelConfig

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm — omits mean-centering and bias terms.
    Computation is promoted to float32 then cast back to the input dtype to
    avoid precision loss with bfloat16 inputs.
    """

    def __init__(self, emb_dim, eps: float = 1e-6):
        """
        Args:
            emb_dim: Size of the last dimension to normalize over.
            eps: Small constant added inside rsqrt for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        """Normalize x by its RMS and scale by the learnable gamma parameter."""
        ms = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(ms + self.eps)
        return (x_normed * self.gamma.float()).type_as(x)
