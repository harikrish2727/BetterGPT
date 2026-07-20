"""
This script trains a tokenizer on the TinyStories dataset and saves it to the specified path.
The tokenizer is trained using the TokenizerTrainer class,
which handles the training process and saves the trained tokenizer to disk.
The script uses the HuggingFace datasets library to load the fineweb,finemath, cosmopedia datasets in streaming mode,
and the tokenizer is trained on a subset of 3,000,000 samples from the dataset.
The tokenizer is configured with a vocabulary size of 32768 and special tokens for
beginning-of-sequence, end-of-sequence, padding, and unknown tokens.
"""

from datasets import load_dataset,interleave_datasets

from src.utils.paths import TOKENIZER_DIR
from src.tokenizer import TokenizerTrainer
from configs.tokenizer_config import TokenizerConfig
from src.utils.logger import get_logger


logger = get_logger(__name__)

path = TOKENIZER_DIR

def is_valid(example):
    return len(example["text"].strip().split()) >= 20


def train_tokenizer(
        sample_size:int,
        vocab_size:int,
        special_tokens:list,
        seed:int,
        output_dir_path
        ):
        """function to train tokenizer on streaming dataset
            args:
            sample_size: total number of samples to train on tokenizer
            vocab_size: total number of vocabulary
            special_tokens: special tokens list,
            seed: randomness,
            output_dir_path: directory path to save tokenizer
        """
        logger.info("Loading datasets...")
        ds_fineweb = load_dataset("HuggingFaceFW/fineweb-edu","sample-10BT",split="train",streaming=True)
        ds_finemath = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
        ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia","auto_math_text",split="train",streaming=True)
        ds_starcoder = load_dataset("bigcode/starcoderdata", data_dir="python", split="train",streaming=True)
        ds_starcoder = ds_starcoder.rename_column("content","text")

        ds_fineweb = ds_fineweb.map(lambda x: {"text": x["text"]})
        ds_finemath = ds_finemath.map(lambda x: {"text": x["text"]})
        ds_cosmo = ds_cosmo.map(lambda x: {"text": x["text"]})
        ds_starcoder = ds_starcoder.map(lambda x :{"text": x["text"]})
        logger.info("datasets loaded")

        dataset = interleave_datasets(
            datasets=[ds_fineweb,ds_cosmo,ds_finemath,ds_starcoder],
            probabilities=[0.7,0.1,0.1,0.1],
            seed=seed,
            )
        logger.info("dataset interleaved")
        dataset = dataset.filter(is_valid)
        dataset = dataset.shuffle(buffer_size=100_000,seed=seed).take(sample_size)
        logger.info("dataset shuffled")
        
        tok = TokenizerTrainer(
            dataset=dataset,
            config=TokenizerConfig(
                vocab_size=vocab_size,
                special_tokens=special_tokens
                )
                )
        logger.info("begin tokenizer training")
        tok.train(output_dir_path)
        logger.info(f"tokenizer trained and saved in {output_dir_path}")



if __name__ == "__main__":
    VOCAB_SIZE = 32768
    special_tokens =  ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",  "<|system|>","<|user|>","<|assistant|>","<|im_start|>","<|im_end|>"]
    logger.info("starting...")

    train_tokenizer(vocab_size=VOCAB_SIZE,
         special_tokens=special_tokens,
         sample_size=3_000_000,
         seed=1227,
         output_dir_path=path)
