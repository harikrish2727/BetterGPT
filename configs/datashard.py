from dataclasses import dataclass
from tokenizers import Tokenizer
from pathlib import Path


from src.paths import TOKENIZER_DIR, DATA_DIR


@dataclass
class DatasetConfig:
    """Hyperparameters settings for dataset shards."""

    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_path: Path = TOKENIZER_DIR
    out_dir: Path = DATA_DIR
    data_column_name: str = "text"
    buffer_size: int = 20_000_000

    def __post_init__(self):
        if self.dataset_name is None or self.data_column_name is "":
            raise ValueError(f" give dataset name")

