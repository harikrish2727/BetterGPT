from transformers import PretrainedConfig


class BetterGPTConfig(PretrainedConfig):
    model_type = "better_gpt"

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
        assert emb_dim % head_count == 0, "emb_dim must be divisible by head_count"
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_blocks = num_blocks
        self.head_count = head_count
        self.rmsnorm_eps = rmsnorm_eps
        self.seq_length = seq_length
        self.ffn_multiple = ffn_multiple
        self.rope_base = rope_base
        self.head_dim = self.emb_dim // self.head_count
        self.architectures = ["BetterGPTModel"]

        # Initialize the Hugging Face base config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

