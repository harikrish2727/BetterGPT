import torch
from src.data_preparation.dataset import TinyDataset
from torch.utils.data import DataLoader
from configs.model import ModelConfig
from configs.training import TrainingConfig
from src.paths import DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)

path = DATA_DIR/"TinyStories_data"
pin_memory = torch.cuda.is_available()
num_workers = 4 if torch.cuda.is_available() else 0

try:
    train_dataset = TinyDataset(
        path=path,
        seq_length=ModelConfig().seq_length,
        split="train",
        infinite=True
    )

    val_dataset = TinyDataset(
        path=path,
        seq_length=ModelConfig().seq_length,
        split="validation",
        infinite=False
    )
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"{e}\n"
        "Data shards not found. Run `python scripts/create_data_shards.py` first."
    ) from e

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=TrainingConfig().batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=pin_memory,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=TrainingConfig().batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=pin_memory,
    drop_last=True
)

logger.info(
    "DataLoaders ready | batch_size=%d | seq_length=%d | workers=%d",
    TrainingConfig().batch_size, ModelConfig().seq_length, num_workers,
)
