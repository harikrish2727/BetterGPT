import math
import torch

import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from model import BetterGPT
from model_config import ModelConfig
from data_loader import train_loader, val_loader


model = BetterGPT(ModelConfig())


device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 3
print(device)
model = model.to(device)


# --- param groups: decay 2D weights, don't decay norms/biases ---
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() >= 2:          # linear/embedding weight matrices
        decay.append(p)
    else:                     # RMSNorm gamma (1D), any biases
        no_decay.append(p)

optim_groups = [
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(
    optim_groups,
    lr=3e-4,                  # peak LR — was 1e-6, ~300x too low
    betas=(0.9, 0.95),        # tuple, LM-standard moments
    eps=1e-8,
)

# total_steps  = num_epochs * len(val_loader)  if using dataset class, here I am using IterableDataset class, which doesn't have __len__ method, so I am calculating total steps based on total tokens and batch size and seq length
total_steps = 470_000_000 / (16 * 64)  #total token/(batch*seq)
warmup_steps = int(0.02 * total_steps)           # ~2%

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


step=0
for i in range(num_epochs):
    model.train()
    epoch_loss = []

    for input_seq, tar_seq in tqdm(val_loader, desc=f"Epoch {i+1}/{num_epochs}"):
        input_seq, tar_seq = input_seq.to(device), tar_seq.to(device)

        # print(input_seq.shape,tar_seq.shape)
        logits = model(input_seq)
        # print(logits.shape)
        B,T,V = logits.shape
        loss = F.cross_entropy(logits.view(B*T,V),tar_seq.view(B*T))
        print(f"loss:{loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        # break

