import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from .transformer_block import TransformerBlock
from .positional_embeddings import RoPESplitHalf
from .layer_normalization import RMSNorm
from .configuration_bettergpt import BetterGPTConfig
from .logger import get_logger

logger = get_logger(__name__)


class BetterGPTModel(PreTrainedModel):
    """BetterGPT model with rotary position embeddings and pre-norm transformer blocks.
    This model is designed for efficient training and inference, supporting gradient checkpointing"""

    config_class = BetterGPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config: BetterGPTConfig):
        super().__init__(config)
        self.gradient_checkpointing = False

        hid = int((8 * config.emb_dim) // 3)
        hid_dim = config.ffn_multiple * (
            (hid + config.ffn_multiple - 1) // config.ffn_multiple
        )
        head_dim = config.emb_dim // config.head_count

        self.emb_layer = nn.Embedding(config.vocab_size, config.emb_dim)
        self.rmsnorm = RMSNorm(config.emb_dim, eps=config.rmsnorm_eps)
        self.rope = RoPESplitHalf(head_dim=head_dim, base=config.rope_base)

        self.transformer_block = nn.ModuleList(
            [
                TransformerBlock(
                    head_count=config.head_count,
                    head_dim=head_dim,
                    emb_dim=config.emb_dim,
                    hid_dim=hid_dim,
                    eps=config.rmsnorm_eps,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = self.emb_layer(input_ids)
        cos, sin = self.rope(x, x.shape[1])

        for block in self.transformer_block:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, sin, cos, attention_mask, use_reentrant=False
                )
            else:
                x = block(x, sin, cos, attention_mask)

        x = self.rmsnorm(x)

        # Base model returns the raw hidden states
        return BaseModelOutputWithPast(last_hidden_state=x)
