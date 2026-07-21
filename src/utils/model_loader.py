import torch

from models.modeling_bettergpt import BetterGPTForCausalLM
from configs.configuration_bettergpt import BetterGPTConfig as ModelConfig
from src.utils.logger import get_logger


logger = get_logger(__name__)

def load_model(saved_model_path, tokenizer):
    model = BetterGPTForCausalLM(ModelConfig())
    ckpt = torch.load(
        saved_model_path, map_location="cpu", weights_only=True
    )
    
    state_dict = ckpt["model_state"]

    model.load_state_dict(state_dict, strict=False)

    logger.info("Pretrained weights loaded successfully!")

    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    logger.info("Resized token embeddings to match tokenizer size.")
    assert model.lm_head.weight.data_ptr() == model.model.emb_layer.weight.data_ptr(), (
        "tie broke after resize"
    )
    logger.info(f"vocab: {model.lm_head.weight.shape[0]} == {len(tokenizer)}")
    return model
