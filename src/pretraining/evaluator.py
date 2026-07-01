import torch
import torch.nn.functional as F

from src.logger import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    device_str = device if isinstance(device, str) else device.type
    model.eval()
    losses = []

    for i, (input_seq, tar_seq) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_seq = input_seq.to(device, non_blocking=True)
        tar_seq = tar_seq.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16, enabled=(device_str == "cuda")):
            logits = model(input_seq)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), tar_seq.view(B * T))
        losses.append(loss.item())

    if not losses:
        logger.warning("Evaluator received zero batches — validation set may be empty.")
        return float("inf")

    avg_loss = sum(losses) / len(losses)
    logger.debug("Evaluated %d batches | avg_val_loss=%.4f", len(losses), avg_loss)
    return avg_loss
