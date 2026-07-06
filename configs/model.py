from transformers import PretrainedConfig


class BetterGPTConfig(PretrainedConfig):
    model_type = "better_gpt"
    attribute_map = {                                           #for trl look up for standard attribute names
        "hidden_size": "emb_dim",
        "num_attention_heads": "head_count",
        "num_hidden_layers": "num_blocks",
        "max_position_embeddings": "seq_length",
    } 

    def __init__(
        self,
        vocab_size=8192,
        emb_dim=512,
        num_blocks=8,
        head_count=8,
        seq_length=512,
        ffn_multiple=128,
        tie_word_embeddings=True,
        rmsnorm_eps=1e-6,
        rope_base=10000,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_blocks = num_blocks
        self.head_count = head_count
        self.rmsnorm_eps = rmsnorm_eps
        self.seq_length = seq_length
        self.ffn_multiple = ffn_multiple
        self.rope_base = rope_base
        self.head_dim = self.emb_dim // self.head_count

        # Initialize the Hugging Face base config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

