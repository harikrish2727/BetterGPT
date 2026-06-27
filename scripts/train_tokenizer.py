
from datasets import load_dataset
from src.paths import TOKENIZER_DIR
from src.tokenizer_train import TokenizerTrainer
from configs.tokenizer_config import TokenizerConfig


path = TOKENIZER_DIR

def main(vocab_size=8192, special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"]):
    ds = load_dataset("roneneldan/TinyStories",split="train",streaming=True).take(300000)
    tok = TokenizerTrainer(
        dataset=ds,
        config=TokenizerConfig(
            vocab_size=8192,
            special_tokens=special_tokens
            )
            )
    tokenizer = tok.train(path)
    return tokenizer


if __name__ == "__main__":
 
    vocab_size = 8192
    special_tokens = ["<bos>","<eos>","<pad>","<unk>"]
    main(vocab_size, special_tokens)