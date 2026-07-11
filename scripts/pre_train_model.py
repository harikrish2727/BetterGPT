import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR


from src.models.model import BetterGPT
from src.pretraining.pretrainer import training
from src.data_preparation.pretrainer_dataloader import (stable_train_loader,
                                                        stable_val_loader,
                                                        logic_train_loader,
                                                        logic_val_loader)
from configs.model import BetterGPTConfig
from configs.training import TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


SEQ_LENGTH = TrainingConfig.seq_length
BATCH_SIZE = TrainingConfig.batch_size
GRAD_ACCUM_STEPS = TrainingConfig.grad_accum_steps

if __name__ == "__main__":
    # 1. Model Initialization
    model_config = BetterGPTConfig()
    model = BetterGPT(model_config)
    model = torch.compile(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model initialized | parameters=%dM", total_params // 1_000_000)

    path = TrainingConfig.checkpoint_dir
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info("Created checkpoint directory: %s", path)

    
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
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=TrainingConfig.learning_rate,
        betas=TrainingConfig.betas,
        eps=TrainingConfig.eps,
        fused=True
    )

    # 3. Calculate Step Boundaries for the WSD Scheduler
    # Tokens processed per actual optimizer step
    tokens_per_step = BATCH_SIZE * SEQ_LENGTH * GRAD_ACCUM_STEPS
    
    TOTAL_STEPS = int(TrainingConfig.target_token_pretraining / tokens_per_step)
    DECAY_START_STEP = int(TrainingConfig.token_switch_threshold / tokens_per_step)
    
    WARM_UP_STEPS = int(0.02 * TOTAL_STEPS)  # 2% Warmup
    DECAY_STEPS = TOTAL_STEPS - DECAY_START_STEP

    logger.info("LR schedule | total_steps=%d | warmup_steps=%d | decay_start=%d", 
                TOTAL_STEPS, WARM_UP_STEPS, DECAY_START_STEP)

    # 4. Custom Warmup-Stable-Decay (WSD) Scheduler
    def wsd_lr_lambda(current_step):
        # Phase 1: Linear Warmup
        if current_step < WARM_UP_STEPS:
            return float(current_step) / float(max(1, WARM_UP_STEPS))
        # Phase 2: Stable Phase (Hold at max LR)
        elif current_step < DECAY_START_STEP:
            return 1.0
        # Phase 3: Linear Decay to 0
        else:
            decay_ratio = float(current_step - DECAY_START_STEP) / float(max(1, DECAY_STEPS))
            return max(0.0, 1.0 - decay_ratio)

    lr_scheduler = LambdaLR(optimizer, lr_lambda=wsd_lr_lambda)

    stable_loaders = (stable_train_loader,stable_val_loader)
    logic_loaders = (logic_train_loader,logic_val_loader)


    # 6. Execution
    try:
        training(
            model=model,
            model_config=model_config,
            stable_loaders=stable_loaders,
            logic_loaders=logic_loaders,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=TrainingConfig.device,
            target_total_tokens=TrainingConfig.target_token_pretraining,
            token_switch_threshold=TrainingConfig.token_switch_threshold,
            save_path=path,
            resume_checkpoint=os.path.join(path,"checkpoint_step_5000.pt"),  # Set this to a path string if resuming else None
            eval_every=TrainingConfig.eval_every,
            save_every=TrainingConfig.save_every
        )
    except Exception as e:
        logger.exception("Training failed with an unhandled exception: %s", e)
        sys.exit(1)