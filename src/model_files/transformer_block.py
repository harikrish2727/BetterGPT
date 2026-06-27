
import torch
import torch.nn as nn

from src.model_files.swiglu_feed_forward import SwiGLU_FFN
from src.model_files.attention import RopeAttention
from src.model_files.layer_normalization import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(
            self,
            emb_dim,
            hid_dim,
            seq_length,
            rope,
            head_count,
            head_dim
            ):

        super().__init__()
        self.pre_attn_norm = RMSNorm(emb_dim)
        self.pre_ffn_norm = RMSNorm(emb_dim)
        self.attention = RopeAttention(
            emb_dim=emb_dim,
            head_dim=head_dim,
            head_count=head_count,
            seq_length=seq_length,
            rope=rope
            )
        self.ffn = SwiGLU_FFN(emb_dim=emb_dim,hid_dim=hid_dim)

    def forward(self,x, attention_mask):
        x = x + self.attention(self.pre_attn_norm(x),attention_mask)
        x = x + self.ffn(self.pre_ffn_norm(x))
        return x