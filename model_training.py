import math
import torch
import os

from dataclasses import asdict
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup





def training(
    model,
    model_config,
    max_steps,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    device,
    save_path,
    resume_checkpoint=None,
    eval_every=500,
):
    step = 0
    best_val_loss = float("inf")
    window_loss = []

    model = model.to(device)

    checkpoint_path=os.path.join(save_path,"checkpoint.pt")
    t_model.pt")best_path=os.path.join(save_path,"bes

    #resume: weights + optimizer + scheduler + step. stream restarts.
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from step {step} (data stream restarts from beginning)")

    def save_ckpt(path):
        tmp = f"{path}.tmp"
        torch.save({
            "step": step,
            "model_config": asdict(model_config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, tmp)
        os.replace(tmp, path)

    model.train()
    train_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, initial=step, desc="Training")

    while step < max_steps:
        input_seq, tar_seq = next(train_iter)
        input_seq, tar_seq = input_seq.to(device, non_blocking=True), tar_seq.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_seq)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), tar_seq.view(B * T))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        window_loss.append(loss.item())
        step += 1
        pbar.update(1)

        if step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            avg_train = sum(window_loss) / len(window_loss)
            print(f"\nStep {step} | train {avg_train:.4f} | val {val_loss:.4f} | "
                  f"grad_norm {grad_norm:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

            save_ckpt(checkpoint_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt(best_path)
                print(f"  best updated: {best_val_loss:.4f}")

            window_loss.clear()
            model.train()


    if window_loss:
        val_loss = evaluate(model, val_loader, device)
        avg_train = sum(window_loss) / len(window_loss)
        print(f"\nFinal step {step} | train {avg_train:.4f} | val {val_loss:.4f}")
        save_ckpt(checkpoint_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(best_path)

    pbar.close()
    return best_val_loss


if __name__ == "__main__":

    from model import BetterGPT
    from config import ModelConfig, TrainingConfig
    from data_loader import train_loader, val_loader
    from model_evaluation import evaluate

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
    
    val_loss = training(
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