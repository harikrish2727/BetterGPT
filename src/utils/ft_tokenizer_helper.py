"""
Prepare tokenizer function is indented to use only if tokenizer is already trained without the special tokens,
and no chat-template is applied. This function add the special tokens and chat template to tokenizer.
"""
from configs.template import chat_template
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_tokenizer(tokenizer):
    """Prepare the tokenizer by adding special tokens and setting the chat template.
    Args:
        tokenizer: The tokenizer to be prepared.
        Returns: The prepared tokenizer.
    """
    additional_special_tokens = [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|im_start|>",
        "<|im_end|>",
    ]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = chat_template
    return tokenizer
