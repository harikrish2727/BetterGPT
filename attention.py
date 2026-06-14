
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

class RopeAttention(nn.Module):
    def __init__(self,emb_dim,head_count,head_dim,rope):
        super().__init__()

        self.head_count = head_count
        self.head_dim = head_dim
        self.rope = rope


        self.qkv_proj = nn.Linear(emb_dim,3*emb_dim,bias=False)

        self.out_proj = nn.Linear(emb_dim,emb_dim,bias=False)


    def forward(self,x):

        batch,seq_length,emb_dim = x.shape
        qkv = self.qkv_proj(x)

        q,k,v = qkv.chunk(3,dim=-1)
        # print(f"q,k,v shape after chunk {q.shape}")

        q = q.view(batch,seq_length,self.head_count,self.head_dim).transpose(1,2)
        k = k.view(batch,seq_length,self.head_count,self.head_dim).transpose(1,2)
        v = v.view(batch,seq_length,self.head_count,self.head_dim).transpose(1,2)

        rotated_q = self.rope(q)
        rotated_k = self.rope(k)
        # print(f"rotated shape q {rotated_q.shape},k {rotated_k.shape}")
        # drop_out_p = self.attn_dropout if self.training else 0.0
        y = scaled_dot_product_attention(rotated_q, rotated_k, v,enable_gqa=True, is_causal=True)

        y = y.transpose(1,2).contiguous()

        y = y.view(batch,seq_length,emb_dim)
        y = self.out_proj(y)
        # print(y.shape)
        return y
