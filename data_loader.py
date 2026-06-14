import torch
from dataset import TinyDataset
from torch.utils.data import DataLoader

path = "./data_shards/TinyStories_data"

pin_memory = torch.cuda.is_available()

train_dataset = TinyDataset(
    path=path,
    seq_length=64,
    split="train",
    infinite=True
    )

val_dataset = TinyDataset(
    path=path,
    seq_length=64,
    split="validation",
    infinite=False
    )

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    num_workers=0,
    pin_memory=pin_memory,
    # persistent_workers=True,
    drop_last=True
    )

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=16,
    num_workers=0,
    pin_memory=pin_memory,
    # persistent_workers=True,
    drop_last=True
    )


