
import torch
import torch.nn as nn


#two ways to create vectorized rope

class RoPE_Interleave(nn.Module):
    def __init__(self, max_seq_len, head_dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        pos = torch.arange(max_seq_len).float()
        # angles = pos[:,None]@inv_freq[None,:]
        angles = torch.outer(pos, inv_freq)

        sin = angles.sin().repeat_interleave(2, dim=-1)
        cos = angles.cos().repeat_interleave(2, dim=-1)

        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
        

    def rotate_half(self, x):
        even = x[..., 0::2]
        odd = x[..., 1::2]

        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def forward(self, x):
        T = x.size(-2)

        sin = self.sin[:T]
        cos = self.cos[:T]
        # print(x.shape)

        return x * cos + self.rotate_half(x) * sin


# class RoPESplitHalf(nn.Module):

#     def __init__(self,head_dim,max_seq_len,base=10000):
#         super().__init__()

#         inv_freq = 1.0 / (
#             base ** (torch.arange(0, head_dim, 2).float() / head_dim)
#         )

#         pos = torch.arange(max_seq_len).float()
#         angles = torch.outer(pos, inv_freq)

#         sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
#         cos = torch.cat([angles.cos(), angles.cos()], dim=-1)

#         self.register_buffer("sin", sin, persistent=False)
#         self.register_buffer("cos", cos, persistent=False)

#     def rotate_half(self,x):
#         x1, x2 = x.chunk(2, dim=-1)
#         return torch.cat((-x2, x1), dim=-1)

#     def forward(self, x):
#         T = x.size(-2)

#         sin = self.sin[:T]
#         cos = self.cos[:T]
#         # print(x.shape)

#         return x * cos + self.rotate_half(x) * sin


class RoPESplitHalf(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)  # persistent: in checkpoint, always loads

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        T = x.size(-2)
        pos = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(pos, self.inv_freq)        # (T, head_dim/2)
        emb = torch.cat([angles, angles], dim=-1)       # (T, head_dim)
        return x * emb.cos() + self.rotate_half(x) * emb.sin()
