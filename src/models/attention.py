
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

class MHAttention(nn.Module):
    def __init__(self,emb_dim,head_count,head_dim,seq_length,rope):
        super().__init__()

        self.head_count = head_count
        self.head_dim = head_dim
        self.rope = rope


        self.qkv_proj = nn.Linear(emb_dim,3*emb_dim,bias=False)

        self.out_proj = nn.Linear(emb_dim,emb_dim,bias=False)

        self.register_buffer("mask",
            torch.tril(torch.ones(seq_length, seq_length)),
            persistent=False
        )


    def forward(self,x,attention_mask):

        batch,seq_length,emb_dim = x.shape
        qkv = self.qkv_proj(x)

        q,k,v = qkv.chunk(3,dim=-1)

        q = q.view(batch,seq_length,self.head_count,self.head_dim).transpose(1,2)
        k = k.view(batch,seq_length,self.head_count,self.head_dim).transpose(1,2)
        v = v.view(batch,seq_length,self.head_count,self.head_dim).transpose(1,2)

        rotated_q = self.rope(q)
        rotated_k = self.rope(k)

        mask = self.mask[:seq_length,:seq_length].bool()

        if attention_mask is not None:
            attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(1)
            combined_mask = attention_mask & mask
            attn_bias = torch.zeros_like(combined_mask, dtype=q.dtype)
            attn_bias.masked_fill_(~combined_mask, torch.finfo(q.dtype).min)
            y = scaled_dot_product_attention(rotated_q,rotated_k,v,attn_mask=attn_bias,is_causal=False)
        else:
            y = scaled_dot_product_attention(rotated_q,rotated_k,v,is_causal=True)

        y = y.transpose(1,2).contiguous()

        y = y.view(batch,seq_length,emb_dim)
        y = self.out_proj(y)
        return y