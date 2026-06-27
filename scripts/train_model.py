import os
import torch

from transformers import get_cosine_schedule_with_warmup

from src.model_files.model import BetterGPT
from configs.config import ModelConfig, TrainingConfig
from src.data_preparation.data_loader import train_loader, val_loader
from src.pre_training.model_training import training

model = BetterGPT(ModelConfig())


path = TrainingConfig().checkpoint_dir

if not os.path.exists(path):
    os.mkdir(path)

#param groups
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() >= 2:          # linear/embedding weight matrices
        decay.append(p)
    else:                     # RMSNorm, gamma (1D), biases
        no_decay.append(p)

optim_groups = [
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(
    optim_groups,
    lr=TrainingConfig().learning_rate,
    betas=TrainingConfig().betas,
    eps=TrainingConfig().eps,
)

# total_steps  = num_epochs * len(val_loader)  if using dataset class, 
# here I am using IterableDataset class, which doesn't have __len__ method, 
# so calculating total steps based on total tokens and batch size and seq length

total_steps = TrainingConfig().token_count / (TrainingConfig().batch_size * ModelConfig().seq_length)  #total token/(batch*seq)
warmup_steps = int(0.02 * total_steps)           # ~2%

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


if __name__ == "__main__":

    
    best_val_loss = training(
        model,
        ModelConfig(),
        total_steps,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        device = TrainingConfig().device,
        save_path=path,
        resume_checkpoint=None,
        eval_every=500
        )
    print(f"model trained, with best validation loss of {best_val_loss}")