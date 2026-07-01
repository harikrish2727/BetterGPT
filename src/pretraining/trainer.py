import math
import torch
import os

from dataclasses import asdict
import torch.nn.functional as F
from tqdm import tqdm

from src.pretraining.evaluator import evaluate
from src.logger import get_logger

logger = get_logger(__name__)


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

    device_str = device if isinstance(device, str) else device.type
    logger.info("Training on %s for %d steps...", device_str, max_steps)
    model = model.to(device)

    checkpoint_path = os.path.join(save_path, "checkpoint.pt")
    best_path = os.path.join(save_path, "best_model.pt")

    # resume: weights + optimizer + scheduler + step. stream restarts.
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        logger.info("Resumed from step %d (data stream restarts from beginning)", step)
    else:
        logger.info("Starting fresh training run")

    def save_ckpt(path):
        tmp = f"{path}.tmp"
        try:
            torch.save({
                "step": step,
                "model_config": asdict(model_config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, tmp)
            os.replace(tmp, path)
        except OSError as e:
            logger.error("Failed to save checkpoint to %s: %s", path, e)

    model.train()
    train_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, initial=step, desc="Training")

    while step < max_steps:
        try:
            input_seq, tar_seq = next(train_iter)
        except StopIteration:
            raise RuntimeError(
                f"Training dataloader exhausted at step {step}/{max_steps}. "
                "Increase token_count or reduce batch_size/seq_length."
            )

        input_seq = input_seq.to(device, non_blocking=True)
        tar_seq = tar_seq.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16, enabled=(device_str == "cuda")):
                logits = model(input_seq)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B * T, V), tar_seq.view(B * T))

            if loss.isnan() or loss.isinf():
                raise RuntimeError(f"Loss is {loss.item()} at step {step} — training has diverged.")

            loss.backward()
        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA out of memory at step %d. Consider reducing batch_size or seq_length.", step
            )
            raise

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        window_loss.append(loss.item())
        step += 1
        pbar.update(1)

        if step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            avg_train = sum(window_loss) / len(window_loss)
            logger.info(
                "Step %d/%d | train_loss=%.4f | val_loss=%.4f | grad_norm=%.4f | lr=%.2e",
                step, max_steps, avg_train, val_loss,
                grad_norm, optimizer.param_groups[0]["lr"],
            )

            save_ckpt(checkpoint_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt(best_path)
                logger.info("New best model saved (val_loss=%.4f)", best_val_loss)

            window_loss.clear()
            model.train()

    if window_loss:
        val_loss = evaluate(model, val_loader, device)
        avg_train = sum(window_loss) / len(window_loss)
        logger.info(
            "Final step %d | train_loss=%.4f | val_loss=%.4f",
            step, avg_train, val_loss,
        )
        save_ckpt(checkpoint_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(best_path)

    pbar.close()
    return best_val_loss
