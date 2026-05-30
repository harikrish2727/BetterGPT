from tokenizer_train import TokenizerConfig, TokenizerTrainer
# from tokenizers import Tokenizer
from datasets import load_dataset


vocab_size = 8192
special_tokens = ["<bos>","<eos>","<pad>","<unk>"]


if __name__ == "__main__":
    ds = load_dataset("roneneldan/TinyStories",split="train",streaming=True).take(300000)
    tok = TokenizerTrainer(dataset=ds,config=TokenizerConfig(vocab_size=8192,special_tokens=special_tokens))
    tokenizer = tok.train("./")
