from dataclasses import dataclass
from pathlib import Path


from src.utils.paths import TOKENIZER_DIR, DATA_DIR

@dataclass
class DatasetConfig:
    """Hyperparameters settings for dataset shards."""

    dataset_name: str
    tokenizer_path: Path = TOKENIZER_DIR
    out_dir: Path = DATA_DIR
    buffer_size: int = 100_000_000

    def __post_init__(self):
        if self.dataset_name is None or self.data_column_name == "":
            raise ValueError(" give dataset name")
