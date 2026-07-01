import os
import sys
import torch

from transformers import get_cosine_schedule_with_warmup

from src.models.model import BetterGPT
from configs.model import ModelConfig
from configs.training import TrainingConfig
from src.data_preparation.data_loader import train_loader, val_loader
from src.pretraining.trainer import training
from src.logger import get_logger

logger = get_logger("train")

model_config = ModelConfig()
training_config = TrainingConfig()

model = BetterGPT(model_config)
total_params = sum(p.numel() for p in model.parameters())
logger.info("Model initialized | parameters=%dM", total_params // 1_000_000)

path = training_config.checkpoint_dir

if not os.path.exists(path):
    os.mkdir(path)
    logger.info("Created checkpoint directory: %s", path)

# param groups: weight decay for matrices, no decay for 1-D params
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() >= 2:
        decay.append(p)
    else:
        no_decay.append(p)

logger.info("Optimizer groups | decay=%d params | no_decay=%d params", len(decay), len(no_decay))

optim_groups = [
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(
    optim_groups,
    lr=training_config.learning_rate,
    betas=training_config.betas,
    eps=training_config.eps,
)

# total_steps = num_epochs * len(val_loader) if using dataset class,
# here using IterableDataset which has no __len__, so derive from token budget
total_steps = int(training_config.token_count / (training_config.batch_size * model_config.seq_length))
warmup_steps = int(0.02 * total_steps)   # ~2%

logger.info("LR schedule | total_steps=%d | warmup_steps=%d", total_steps, warmup_steps)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


if __name__ == "__main__":
    try:
        best_val_loss = training(
            model,
            model_config,
            total_steps,
            train_loader,
            val_loader,
            optimizer,
            lr_scheduler,
            device=training_config.device,
            save_path=path,
            resume_checkpoint=None,
            eval_every=500,
        )
        logger.info("Training complete | best_val_loss=%.4f", best_val_loss)
    except Exception:
        logger.exception("Training failed with an unhandled exception")
        sys.exit(1)
