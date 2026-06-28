import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from src.model_files.transformer_block import TransformerBlock
from src.model_files.positional_embeddings import RoPESplitHalf
from src.model_files.layer_normalization import RMSNorm


from transformers import PretrainedConfig, PreTrainedModel,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast


class BetterGPTConfig(PretrainedConfig):
    model_type = "better_gpt" # Required for HF registries

    def __init__(
        self,
        vocab_size=8192,
        emb_dim=512,
        num_blocks=8,
        head_count=8,
        seq_length=512,
        ffn_multiple=128,
        tie_word_embeddings=True,
        **kwargs
    ):
        assert emb_dim % head_count == 0, "emb_dim must be divisible by head_count"
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_blocks = num_blocks
        self.head_count = head_count
        self.seq_length = seq_length
        self.ffn_multiple = ffn_multiple
        
        # Initialize the Hugging Face base config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)



class BetterGPTModel(PreTrainedModel):
    config_class = BetterGPTConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: BetterGPTConfig):
        super().__init__(config)
        self.gradient_checkpointing = False

        hid = int((8 * config.emb_dim) // 3)
        hid_dim = config.ffn_multiple * ((hid + config.ffn_multiple - 1) // config.ffn_multiple)
        head_dim = config.emb_dim // config.head_count
        self.seq_length = config.seq_length

        rope = RoPESplitHalf(head_dim=head_dim, max_seq_len=config.seq_length)

        self.emb_layer = nn.Embedding(config.vocab_size, config.emb_dim)
        self.rmsnorm = RMSNorm(config.emb_dim)

        self.transformer_block = nn.ModuleList([
            TransformerBlock(
                head_count=config.head_count,
                head_dim=head_dim,
                emb_dim=config.emb_dim,
                hid_dim=hid_dim,
                seq_length=self.seq_length,
                rope=rope
            ) for _ in range(config.num_blocks)
        ])


    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = self.emb_layer(input_ids)
        for block in self.transformer_block:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, attention_mask, use_reentrant=False
                )
            else:
                x = block(x, attention_mask)
        
        x = self.rmsnorm(x)
        
        # Base model returns the raw hidden states
        return BaseModelOutputWithPast(last_hidden_state=x)



class BetterGPT(PreTrainedModel,GenerationMixin):
    """
    A small decoder-only language model adapted for Hugging Face Trainer compatibility.
    """
    
    config_class = BetterGPTConfig 
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.emb_layer.weight"}
    supports_gradient_checkpointing = True
    def __init__(self, config: BetterGPTConfig):
        # Call the HF PreTrainedModel init
        super().__init__(config) 
        

        self.gradient_checkpointing = False
        
        self.model = BetterGPTModel(config)
        self.seq_length = config.seq_length
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        # self.apply(self._init_weights)  hf post_init() calls this automatically
        self.post_init()

        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_blocks))

        # self.lm_head.weight = self.model.emb_layer.weight    # weight tying

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

    #two methods, so resize_token_embeddings() works
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

    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        logits = self.lm_head(hidden_states)

        # Calculate the loss automatically if labels are provided (Trainer requires this)
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

        # Return a Hugging Face CausalLMOutputWithPast object
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    @torch.no_grad()
    def custom_generate(self, idx, max_tokens, temp, top_k=None, eos_id=None):
        self.eval()
        B = idx.size(0)
        finished = torch.zeros(B, 1, dtype=torch.bool, device=idx.device)
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.seq_length:]
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
                    logits = logits.masked_fill(logits < min_val, float('-inf'))
                probs = F.softmax(logits, dim=-1)
                out_idx = torch.multinomial(probs, num_samples=1)

            if eos_id is not None:
                out_idx = torch.where(finished, torch.full_like(out_idx, eos_id), out_idx)
                finished = finished | (out_idx == eos_id)

            idx = torch.cat([idx, out_idx], dim=1)

            if eos_id is not None and finished.all():
                break

        return idx