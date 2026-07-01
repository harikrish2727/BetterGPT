import os
import numpy as np
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer

from src.logger import get_logger

logger = get_logger(__name__)


class ShardDataset:
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Tokenizer,
        out_dir: str,
        split: str = "train",
        data_column_name: str = "text",
        buffer_size: int = 20_000_000
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        data_folder_name = f"{dataset_name.split('/')[-1]}_data"
        self.data_column_name = data_column_name
        self.out_dir = os.path.join(out_dir, data_folder_name)
        self.buffer_size = buffer_size

        self.vocab_size = int(tokenizer.get_vocab_size())
        os.makedirs(self.out_dir, exist_ok=True)

        if self.vocab_size >= 65000:
            raise ValueError(
                f"vocab_size={self.vocab_size} exceeds 64999; cannot store token IDs in uint16."
            )

    def save_shards(self, data_ids, shard_name):
        shard_path = os.path.join(self.out_dir, shard_name)
        if not os.path.exists(shard_path):
            data_ids.tofile(shard_path)
            logger.info("Saved shard: %s", shard_name)

    def run(self, split):
        shard_id = 1
        buffer = np.empty(self.buffer_size, dtype=np.uint16)
        buffer_idx = 0

        ds = load_dataset(self.dataset_name, split=split, streaming=True).shuffle(seed=42, buffer_size=10_000)

        for samples in ds.iter(1000):
            encoded = self.tokenizer.encode_batch(samples[self.data_column_name])

            for item in encoded:
                ids = np.array(item.ids, dtype=np.uint16)
                start = 0

                while start < len(ids):
                    remaining = self.buffer_size - buffer_idx
                    take = min(remaining, len(ids) - start)

                    buffer[buffer_idx:buffer_idx + take] = ids[start:start + take]

                    buffer_idx += take
                    start += take

                    if buffer_idx == self.buffer_size:
                        shard_name = f"{split}_shard{shard_id:04d}.bin"
                        self.save_shards(buffer, shard_name)

                        if shard_id % 5 == 0:
                            logger.info("Progress: %d shards created", shard_id)

                        shard_id += 1
                        buffer_idx = 0

        if buffer_idx > 0:
            shard_name = f"{split}_shard{shard_id:04d}.bin"
            self.save_shards(buffer[:buffer_idx], shard_name)
