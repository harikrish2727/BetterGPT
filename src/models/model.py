import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model import ModelConfig
from src.models.transformer_block import TransformerBlock
from src.models.positional_embeddings import RoPESplitHalf
from src.models.layer_normalization import RMSNorm
from src.logger import get_logger

logger = get_logger(__name__)


class BetterGPT(nn.Module):
    """A small decoder only langauge model inspired from architeture of GPT and LLama models,
       packed with RoPE positional embedding, flash attention for optimized attention computation with
       grouped query attention, and used fused kernals and weight tying.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()

        hid = int((8 * config.emb_dim) // 3)
        hid_dim = config.ffn_multiple * ((hid + config.ffn_multiple - 1) // config.ffn_multiple)
        head_dim = config.emb_dim // config.head_count
        self.seq_length = config.seq_length

        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim, got {head_dim} "
                f"(emb_dim={config.emb_dim}, head_count={config.head_count})"
            )

        rope = RoPESplitHalf(
            head_dim=head_dim,
            max_seq_len=config.seq_length
        )

        self.emb_layer = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.emb_dim
        )

        self.rmsnorm = RMSNorm(config.emb_dim)

        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self.transformer_block = nn.ModuleList([TransformerBlock(
            head_count=config.head_count,
            head_dim=head_dim,
            emb_dim=config.emb_dim,
            hid_dim=hid_dim,
            seq_length=self.seq_length,
            rope=rope
        ) for _ in range(config.num_blocks)])

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_blocks))

        self.lm_head.weight = self.emb_layer.weight    # weight tying

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_dim, attention_mask=None):
        x = self.emb_layer(input_dim)
        for block in self.transformer_block:
            x = block(x, attention_mask)

        x = self.rmsnorm(x)
        x = self.lm_head(x)

        return x

    @torch.no_grad()
    def generate(self, idx, max_tokens, temp, top_k=None, eos_id=None):
        if idx.dim() != 2:
            raise ValueError(f"generate() expects a 2-D input (B, T), got shape {tuple(idx.shape)}")
        if temp < 0:
            raise ValueError(f"temp must be >= 0, got {temp}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")

        self.eval()
        B = idx.size(0)
        finished = torch.zeros(B, 1, dtype=torch.bool, device=idx.device)

        for _ in range(max_tokens):
            if idx.size(1) > self.seq_length:
                logger.warning(
                    "Input length %d exceeds seq_length %d; truncating to last %d tokens.",
                    idx.size(1), self.seq_length, self.seq_length,
                )
            idx_cond = idx[:, -self.seq_length:]
            logits = self(idx_cond)[:, -1, :]

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise RuntimeError("NaN or Inf detected in model logits — check model weights.")

            if temp == 0:
                out_idx = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temp
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    val, _ = torch.topk(logits, k)
                    min_val = val[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < min_val, float('-inf'))
                probs = F.softmax(logits, dim=-1)
                out_idx = torch.multinomial(probs, num_samples=1)

            if eos_id is not None:
                # rows already done keep emitting EOS (clean padding, not real tokens)
                out_idx = torch.where(finished, torch.full_like(out_idx, eos_id), out_idx)
                finished = finished | (out_idx == eos_id)

            idx = torch.cat([idx, out_idx], dim=1)

            if eos_id is not None and finished.all():
                break

        return idx

    @torch.no_grad()
    def generate_single(self, idx, max_tokens, temp, top_k=None, eos_id=None):
        if idx.dim() != 1:
            raise ValueError(f"generate_single() expects a 1-D input (T,), got shape {tuple(idx.shape)}")
        if temp < 0:
            raise ValueError(f"temp must be >= 0, got {temp}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")

        self.eval()
        for _ in range(max_tokens):
            idx_cond = idx[-self.seq_length:]
            logits = self(idx_cond.unsqueeze(0))[:, -1, :]

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise RuntimeError("NaN or Inf detected in model logits — check model weights.")

            if temp == 0:
                out_idx = torch.argmax(logits, dim=-1).item()
            else:
                logits = logits / temp
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    val, _ = torch.topk(logits, k)
                    min_val = val[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < min_val, float('-inf'))
                probs = F.softmax(logits, dim=-1)
                out_idx = torch.multinomial(probs, num_samples=1).item()

            if eos_id is not None and out_idx == eos_id:
                break

            idx = torch.cat([idx, torch.tensor([out_idx], device=idx.device)], dim=0)
        return idx
