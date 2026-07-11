import torch

from torch.utils.data import DataLoader

from configs.training import TrainingConfig
from src.utils.logger import get_logger
from src.utils.paths import DATA_DIR
from src.data_preparation.dataset import PreTrainingDataset

logger = get_logger(__name__)

PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

STABLE_TRAIN_DIR = DATA_DIR/"stable/train"
STABLE_VAL_DIR = DATA_DIR/"stable/valid"

ANNEAL_TRAIN_DIR = DATA_DIR/"logic/train"
ANNEAL_VAL_DIR = DATA_DIR/"logic/valid"

#CONFIGURATION


SEQ_LENGTH = TrainingConfig.seq_length
BATCH_SIZE = TrainingConfig.batch_size


#stable phase

stable_train_dataset = PreTrainingDataset(STABLE_TRAIN_DIR, SEQ_LENGTH, infinite=True)
stable_valid_dataset = PreTrainingDataset(STABLE_VAL_DIR, SEQ_LENGTH, infinite=False)

#valid_phase

annealing_train_dataset = PreTrainingDataset(ANNEAL_TRAIN_DIR, SEQ_LENGTH, infinite=True)
annealing_val_dataset = PreTrainingDataset(ANNEAL_VAL_DIR, SEQ_LENGTH, infinite=False)


#DATALOADERS


stable_train_loader = DataLoader(
                                    stable_train_dataset,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    persistent_workers=PIN_MEMORY,
                                    drop_last=True
                                )
stable_val_loader   = DataLoader(
                                    stable_valid_dataset,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    persistent_workers=PIN_MEMORY,
                                    drop_last=True
                                )

logic_train_loader  = DataLoader(
                                    annealing_train_dataset,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    persistent_workers=PIN_MEMORY,
                                    drop_last=True
                                )

logic_val_loader    = DataLoader(
                                    annealing_val_dataset,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=PIN_MEMORY,
                                    persistent_workers=PIN_MEMORY,
                                    drop_last=True
                                )

