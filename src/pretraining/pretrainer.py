import os
import torch

from dataclasses import asdict
import torch.nn.functional as F
from tqdm import tqdm

from configs.training import TrainingConfig
from src.pretraining.evaluator import evaluate
from src.utils.logger import get_logger
from src.data_preparation.pretrainer_dataloader import (stable_train_loader,
                                                        stable_val_loader,
                                                        logic_train_loader,
                                                        logic_val_loader
                                                        )

logger = get_logger(__name__)

SEQ_LENGTH = TrainingConfig.seq_length
BATCH_SIZE = TrainingConfig.batch_size

GRAD_ACCUM_STEPS = TrainingConfig.grad_accum_steps
# TOKEN_SWITCH_THRESHOLD = TrainingConfig.token_switch_threshold  # 9 Billion tokens
# TARGET_TOTAL_TOKENS = TrainingConfig.target_token_pretraining    # 10 Billion tokens


def fast_forward_iterator(iterator, batches_to_skip):
    """Skips ahead in an iterable dataset to prevent data repetition upon resuming."""
    if batches_to_skip > 0:
        logger.info(f"Fast-forwarding {batches_to_skip} batches... this may take a few minutes.")
        for _ in range(batches_to_skip):
            try:
                next(iterator)
            except StopIteration:
                pass
        logger.info("Fast-forward complete!")
    return iterator

def training(
    model,
    model_config,
    stable_loaders, # Tuple: (train, val)
    logic_loaders,  # Tuple: (train, val)
    optimizer,
    lr_scheduler,
    device,
    save_path,
    target_total_tokens,
    token_switch_threshold,
    resume_checkpoint=None,
    eval_every=500,
    save_every=2000, # Save historical checkpoints periodically
):
    device_str = device if isinstance(device, str) else device.type
    model = model.to(device)

    # State variables
    step = 0
    global_tokens = 0
    using_logic_data = False
    window_loss = []

    # Unpack Loaders
    stable_train, stable_val = stable_loaders
    logic_train, logic_val = logic_loaders
    
    
    active_train_iter = iter(stable_train)
    active_val_loader = stable_val

    # --- RESUME LOGIC ---
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        step = ckpt["step"]
        global_tokens = ckpt["global_tokens"]
        
        # Check which phase we are in to set the correct iterator
        if global_tokens >= token_switch_threshold:
            using_logic_data = True
            active_train_iter = iter(logic_train)
            active_val_loader = logic_val
            
        logger.info(f"Resumed from step {step} ({global_tokens/1e9:.3f}B tokens)")
        
        # Fast-forward the active dataloader
        # multiply by GRAD_ACCUM_STEPS because 'step' tracks optimizer steps, 
        # but the dataloader yields 'micro-batches'.
        batches_to_skip = step * GRAD_ACCUM_STEPS
        active_train_iter = fast_forward_iterator(active_train_iter, batches_to_skip)
    else:
        logger.info("Starting fresh training run")

    def save_ckpt(name):
        path = os.path.join(save_path, f"{name}.pt")
        tmp = f"{path}.tmp"
        try:
            torch.save({
                "step": step,
                "global_tokens": global_tokens,
                "model_config": asdict(model_config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
            }, tmp)
            os.replace(tmp, path)
        except OSError as e:
            logger.error(f"Failed to save checkpoint to {path}: {e}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    # Calculate total steps for the progress bar
    estimated_tokens_per_step = BATCH_SIZE * SEQ_LENGTH * GRAD_ACCUM_STEPS
    total_steps = target_total_tokens // estimated_tokens_per_step
    pbar = tqdm(total=total_steps, initial=step, desc="Training")

    # MAIN LOOP
    while global_tokens < target_total_tokens:
        
        # 1. PHASE SWITCH LOGIC
        if global_tokens >= token_switch_threshold and not using_logic_data:
            logger.info(f"--- 90% TOKENS REACHED. SWAPPING TO LOGIC DATA ---")
            active_train_iter = iter(logic_train)
            active_val_loader = logic_val
            using_logic_data = True

        # 2. FETCH DATA
        try:
            input_seq, tar_seq = next(active_train_iter)
        except StopIteration:
            # Re-initialize the active iterator if a physical shard ends
            active_train_iter = iter(logic_train if using_logic_data else stable_train)
            input_seq, tar_seq = next(active_train_iter)

        input_seq = input_seq.to(device, non_blocking=True)
        tar_seq = tar_seq.to(device, non_blocking=True)
        
        # Update token tracker precisely
        global_tokens += input_seq.numel()

        # 3. FORWARD & BACKWARD PASS
        with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16, enabled=(device_str == "cuda")):
            logits = model(input_seq).logits
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), tar_seq.view(B * T))
            # Scale loss for gradient accumulation
            loss = loss / GRAD_ACCUM_STEPS

        if loss.isnan() or loss.isinf():
            raise RuntimeError(f"Loss is {loss.item()} at step {step} — training diverged.")

        loss.backward()
        window_loss.append(loss.item() * GRAD_ACCUM_STEPS) # Unscale for logging
        micro_step += 1

        # 4. OPTIMIZER STEP (Gradient Accumulation Execution)
        if micro_step % GRAD_ACCUM_STEPS == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            step += 1
            pbar.update(1)

            # --- EVALUATION ---
            if step % eval_every == 0:
                val_loss = evaluate(model, active_val_loader, device)
                avg_train = sum(window_loss) / len(window_loss)
                phase_name = "LOGIC" if using_logic_data else "STABLE"
                
                logger.info(
                    f"Step {step} | Phase: {phase_name} | Tokens: {global_tokens/1e9:.3f}B | "
                    f"Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                
                window_loss.clear()
                model.train() # Return to training mode after eval
                
                # Always save the latest state
                save_ckpt("checkpoint_latest")

            # --- HISTORICAL CHECKPOINTING ---
            if step % save_every == 0:
                save_ckpt(f"checkpoint_step_{step}")

    pbar.close()
    
    # Final save and eval at 10B tokens
    final_val = evaluate(model, active_val_loader, device)
    logger.info(f"Training Complete! Final Val Loss: {final_val:.4f}")
    save_ckpt("checkpoint_final")