
import torch
import torch.nn as nn

from swiglu_feed_forward import SwiGLU_FFN
from attention import RopeAttention
from layer_normalization import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self,emb_dim,hid_dim,rope,head_count,head_dim):
        super().__init__()
        self.pre_attn_norm = RMSNorm(emb_dim)
        self.pre_ffn_norm = RMSNorm(emb_dim)
        self.attention = RopeAttention(emb_dim=emb_dim,head_dim=head_dim,head_count=head_count,rope=rope)
        self.ffn = SwiGLU_FFN(emb_dim=emb_dim,hid_dim=hid_dim)

    def forward(self,x):
        x = x + self.attention(self.pre_attn_norm(x))
        x = x + self.ffn(self.pre_ffn_norm(x))
        return x
