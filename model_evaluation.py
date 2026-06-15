import torch
import torch.nn.functional as F

@torch.no_grad()                     
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()                      
    losses = []
    for i, (input_seq, tar_seq) in enumerate(val_loader):
        if i >= max_batches:        
            break
        input_seq = input_seq.to(device, non_blocking=True)
        tar_seq   = tar_seq.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_seq)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), tar_seq.view(B * T))
        losses.append(loss.item())
    return sum(losses) / len(losses)