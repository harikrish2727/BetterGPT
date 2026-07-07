"""
This script trains a tokenizer on the TinyStories dataset and saves it to the specified path.
The tokenizer is trained using the TokenizerTrainer class,
which handles the training process and saves the trained tokenizer to disk.
The script uses the HuggingFace datasets library to load the TinyStories dataset in streaming mode,
and the tokenizer is trained on a subset of 300,000 samples from the dataset.
The tokenizer is configured with a vocabulary size of 8192 and special tokens for
beginning-of-sequence, end-of-sequence, padding, and unknown tokens.
"""

from datasets import load_dataset

from src.utils.paths import TOKENIZER_DIR
from src.tokenizer import TokenizerTrainer
from configs.tokenizer_config import TokenizerConfig
from src.utils.logger import get_logger


logger = get_logger(__name__)

path = TOKENIZER_DIR


def main(vocab_size: int, special_tokens: list, dataset_name:str,sample_size:int):
    """
    Main function to train a tokenizer on the TinyStories dataset and save it to disk.
    Args:
        vocab_size: The size of the vocabulary for the tokenizer.
        special_tokens: A list of special tokens to include in the tokenizer.
    Returns:
        The trained tokenizer instance.
    """
    logger.info("Loading dataset...")
    ds = load_dataset(dataset_name, split="train", streaming=True).take(
        sample_size
    )
    logger.info(f"{dataset_name} dataset loaded")

    tok = TokenizerTrainer(
        dataset=ds,
        config=TokenizerConfig(vocab_size=vocab_size, special_tokens=special_tokens),
    )
    tok.train(path)
    logger.info("tokenizer trained")


if __name__ == "__main__":
    VOCAB_SIZE = 32768
    special_tokens =  ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",  "<|system|>","<|user|>","<|assistant|>","<|im_start|>","<|im_end|>"]
    logger.info("starting...")

    main(vocab_size=VOCAB_SIZE,
         special_tokens=special_tokens,
         dataset_name="HuggingFaceFW/fineweb-edu",
         sample_size=300)
