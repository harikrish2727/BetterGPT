import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from configs.model import BetterGPTConfig
from src.models.base_model import BetterGPTModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BetterGPT(PreTrainedModel, GenerationMixin):
    """
    A small decoder-only language model adapted for Hugging Face Trainer compatibility.
    """

    config_class = BetterGPTConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.emb_layer.weight"}
    supports_gradient_checkpointing = True

    def __init__(self, config: BetterGPTConfig):
        super().__init__(config)

        self.gradient_checkpointing = False

        self.model = BetterGPTModel(config)
        self.seq_length = config.seq_length
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        # self.apply(self._init_weights)  hf post_init() calls this automatically
        self.post_init()

        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_blocks)
                )

    def _init_weights(self, module):
        """weight initialization for Linear and embedding layers"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _tie_weights(self):
        self.lm_head.weight = self.model.emb_layer.weight

    # TRL access the base model
    def get_decoder(self):
        return self.model

    # two methods to make resize_token_embeddings() work
    def get_input_embeddings(self):
        return self.model.emb_layer

    def set_input_embeddings(self, value):
        self.model.emb_layer = value
        # Re-tie weights after resizing
        self.lm_head.weight = self.model.emb_layer.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # When use_cache=False, just return the inputs as-is
        # The model will process the full sequence each time

        if not kwargs.get("use_cache", True):
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass for the BetterGPT model.
        args:
            input_ids: Tensor of shape (B, T) containing token IDs.
            attention_mask: Optional tensor of shape (B, T) indicating which tokens to attend to.
            labels: Optional tensor of shape (B, T) containing target token IDs for loss computation
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        logits = self.lm_head(hidden_states)

        # to Calculate the loss,if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Return Hugging Face CausalLMOutputWithPast object
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    @torch.no_grad()
    def custom_generate(self, idx, max_tokens, temp, top_k=None, eos_id=None):
        """
        Custom generation loop for the BetterGPT model.
        Args:
            idx: Tensor of shape (B, T) containing initial token IDs.
            max_tokens: Maximum number of tokens to generate.
            temp: Sampling temperature; 0 for greedy decoding.
            top_k: Optional integer for top-k sampling.
            eos_id: Optional ID of the end-of-sequence token.
        """
        self.eval()
        B = idx.size(0)
        finished = torch.zeros(B, 1, dtype=torch.bool, device=idx.device)
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.seq_length :]
            # changed forward, so extract logits from the returned object
            logits = self(input_ids=idx_cond).logits[:, -1, :]

            if temp == 0:
                out_idx = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temp
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    val, _ = torch.topk(logits, k)
                    min_val = val[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < min_val, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                out_idx = torch.multinomial(probs, num_samples=1)

            if eos_id is not None:
                out_idx = torch.where(
                    finished, torch.full_like(out_idx, eos_id), out_idx
                )
                finished = finished | (out_idx == eos_id)

            idx = torch.cat([idx, out_idx], dim=1)

            if eos_id is not None and finished.all():
                break

        return idx
