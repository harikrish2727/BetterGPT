import logging
from datasets import load_dataset
from src.paths import TOKENIZER_DIR
from src.tokenizer import TokenizerTrainer
from configs.tokenizer_config import TokenizerConfig
from src.logger import get_logger


logger = get_logger(__name__)

path = TOKENIZER_DIR

def main(vocab_size=8192, special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"]):
    logger.info("Loading dataset...")
    ds = load_dataset("roneneldan/TinyStories",split="train",streaming=True).take(300000)
    logger.info("TinyStories dataset loaded")

    tok = TokenizerTrainer(
        dataset=ds,
        config=TokenizerConfig(
            vocab_size=8192,
            special_tokens=special_tokens
            )
            )
    tokenizer = tok.train(path)
    logger.info("tokenizer trained")
    return tokenizer


if __name__ == "__main__":
 
    vocab_size = 8192
    special_tokens = ["<bos>","<eos>","<pad>","<unk>"]
    logger.info("starting...")
    main(vocab_size, special_tokens)