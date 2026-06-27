import torch
from src.data_preparation.dataset import TinyDataset
from torch.utils.data import DataLoader
from configs.config import TrainingConfig, ModelConfig
from src.paths import DATA_DIR

path = DATA_DIR

pin_memory = torch.cuda.is_available()

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

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=TrainingConfig().batch_size,
    num_workers=4 if torch.cuda.is_available() else 0,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True if torch.cuda.is_available() else False,
    drop_last=True
    )

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=TrainingConfig().batch_size,
    num_workers=4 if torch.cuda.is_available() else 0,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True if torch.cuda.is_available() else False,
    drop_last=True
    )


