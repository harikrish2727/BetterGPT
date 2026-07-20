import torch.nn as nn
import torch.nn.functional as F


class SwiGLU_FFN(nn.Module):
    """SwiGLU feed-forward network from PaLM / LLaMA.

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x)).
    All three linear layers are bias-free.
    """

    def __init__(self, emb_dim, hid_dim):
        """
        Args:
            emb_dim: Input and output dimension.
            hid_dim: Hidden (expanded) dimension; typically ~(8/3)*emb_dim rounded up
                     to the nearest ffn_multiple.
        """
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim, hid_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hid_dim, bias=False)
        self.down_proj = nn.Linear(hid_dim, emb_dim, bias=False)

    def forward(self, x):
        """Apply the SwiGLU gating: silu(gate) * up, then project back down."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down
