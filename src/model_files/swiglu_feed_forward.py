
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU_FFN(nn.Module):
    def __init__(self, emb_dim,hid_dim):
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim,hid_dim,bias=False)
        self.up_proj = nn.Linear(emb_dim,hid_dim,bias=False)
        self.down_proj = nn.Linear(hid_dim,emb_dim,bias=False)

    def forward(self,x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate*up)

        return down
