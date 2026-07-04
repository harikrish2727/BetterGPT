import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of the input tensor by splitting it in half and negating the second half."""
    x1, x2 = x.chunk(2, dim=-1)
    logger.debug(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(
        q: torch.Tensor, 
        k: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to the query and key tensors."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    logger.debug(f"cos shape: {cos.shape}, sin shape: {sin.shape}")
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    logger.debug(f"q_rot shape: {q_rot.shape}, k_rot shape: {k_rot.shape}")
    return q_rot, k_rot