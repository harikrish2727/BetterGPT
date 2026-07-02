from src.finetuning.template import chat_template


def prepare_tokenizer(tokenizer):
    """Prepare the tokenizer by adding special tokens and setting the chat template.
    Args:
        tokenizer: The tokenizer to be prepared.
        Returns: The prepared tokenizer.
    """
    additional_special_tokens = ["<|system|>","<|user|>","<|assistant|>","<|im_start|>","<|im_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens":additional_special_tokens})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = chat_template
    return tokenizer