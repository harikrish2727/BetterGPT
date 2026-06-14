import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,emb_dim,eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        ms = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(ms + self.eps)
        return (x_normed * self.gamma.float()).type_as(x)
