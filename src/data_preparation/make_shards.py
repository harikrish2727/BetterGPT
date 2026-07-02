import os
from pathlib import Path
import numpy as np
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from src.utils.logger import get_logger
from configs.datashard import DatasetConfig
from configs.tokenizer_config import TokenizerConfig

logger = get_logger(__name__)


class ShardDataset:
    """Tokenize a HuggingFace streaming dataset and write it to binary shard files.

    Token IDs are stored as uint16, limiting vocab_size to < 65 000.
    Each shard holds buffer_size tokens packed contiguously; the final shard may
    be smaller. Shards that already exist on disk are skipped.
    """

    def __init__(
        self,
        dataset_name: str = DatasetConfig.dataset_name,
        tokenizer_path: Tokenizer= DatasetConfig.tokenizer_path,
        out_dir: Path = DatasetConfig.out_dir,
        data_column_name: str = DatasetConfig.data_column_name,
        buffer_size: int = DatasetConfig.buffer_size,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset identifier (e.g. 'roneneldan/TinyStories').
            tokenizer: Trained tokenizers.Tokenizer instance.
            out_dir: Root output directory; shards are written under <out_dir>/<dataset>_data/.
            split: Dataset split to process ('train', 'validation', etc.).
            data_column_name: Column name that contains the raw text strings.
            buffer_size: Number of tokens to accumulate before flushing one shard file.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.dataset_name = dataset_name
        data_folder_name = f"{dataset_name.split('/')[-1]}_data"
        self.data_column_name = data_column_name
        self.out_dir = os.path.join(out_dir, data_folder_name)
        self.buffer_size = buffer_size

        self.vocab_size = int(TokenizerConfig.vocab_size)
        os.makedirs(self.out_dir, exist_ok=True)

        if self.vocab_size >= 65000:
            raise ValueError(
                f"vocab_size={self.vocab_size} exceeds 64999; cannot store token IDs in uint16."
            )

    def save_shards(self, data_ids, shard_name):
        """Write data_ids to a binary shard file, skipping if the file already exists."""
        shard_path = os.path.join(self.out_dir, shard_name)
        if not os.path.exists(shard_path):
            data_ids.tofile(shard_path)
            logger.info("Saved shard: %s", shard_name)

    def run(self, split):
        """Tokenize the given dataset split and write token IDs to shard files.

        Args:
            split: Dataset split name to process ('train', 'validation', etc.).
        """
        shard_id = 1
        buffer = np.empty(self.buffer_size, dtype=np.uint16)
        buffer_idx = 0
        
        try:
            ds = load_dataset(self.dataset_name, split=split, streaming=True).shuffle(seed=42, buffer_size=10_000)
            logger.info("{self.dataset_name} dataset downloaded")
        except Exception as e:
            raise RuntimeError("dataset download failed!!!")

        for samples in ds.iter(1000):
            encoded = self.tokenizer(samples[self.data_column_name])

            for item_ids in encoded["input_ids"]:
                ids = np.array(item_ids, dtype=np.uint16)
                start = 0

                while start < len(ids):
                    remaining = self.buffer_size - buffer_idx
                    take = min(remaining, len(ids) - start)

                    buffer[buffer_idx:buffer_idx + take] = ids[start:start + take]

                    buffer_idx += take
                    start += take

                    if buffer_idx == self.buffer_size:
                        shard_name = f"{split}_shard{shard_id:04d}.bin"
                        try:
                            self.save_shards(buffer, shard_name)
                        except Exception as e:
                            raise RuntimeError(f"data saving failed!!!")

                        if shard_id % 5 == 0:
                            logger.info("Progress: %d shards created", shard_id)

                        shard_id += 1
                        buffer_idx = 0
        
        if buffer_idx > 0:
            shard_name = f"{split}_shard{shard_id:04d}.bin"
            try:
                self.save_shards(buffer[:buffer_idx], shard_name)
                logger.info("final shard saved")
            except Exception as e:
                logger.error("saving last shard failed")
                raise RuntimeError("final shard saving failed")
