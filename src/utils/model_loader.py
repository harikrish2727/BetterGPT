import torch

from src.models.model import BetterGPT
from configs.model import BetterGPTConfig as ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(saved_model_path,tokenizer):
    """this function will safely load the trained model for fine tuning after pretraining. 
    It will remap the state dict to match the new model structure and resize the token embeddings to match the tokenizer size.
    """
    model = BetterGPT(ModelConfig())
    ckpt = torch.load(saved_model_path, map_location="cpu", weights_only=True)  #sft trainer needs cpu, it will load to gpu later
    old_state_dict = ckpt["model_state"]

    new_state_dict = {}
    for key, value in old_state_dict.items():
        if "rope" in key:                      
            continue
        if key.startswith("lm_head"):
            new_state_dict[key] = value         # lm_head stays top-level
        else:
            new_state_dict["model." + key] = value 

    result = model.load_state_dict(new_state_dict, strict=False)

    logger.info("Pretrained weights loaded successfully!")
    
    allowed_missing = {k for k in result.missing_keys if "rope" in k}
    real_missing = set(result.missing_keys) - allowed_missing
    assert not real_missing, f"Real weights missing: {real_missing}"
    assert not result.unexpected_keys, f"Remap wrong: {result.unexpected_keys}"
    logger.info(f"Loaded. Skipped {len(allowed_missing)} rope buffers.")

    model.resize_token_embeddings(len(tokenizer),mean_resizing=True)
    logger.info("Resized token embeddings to match tokenizer size.")
    assert model.lm_head.weight.data_ptr() == model.model.emb_layer.weight.data_ptr(), "tie broke after resize"
    logger.info(f"vocab: {model.lm_head.weight.shape[0]} == {len(tokenizer)}")
    return model

if __name__ == "__main__":
    tokenizer_path = "./tokenizer_checkpoint"
    model_path = "./checkpoints/model.pt"
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    model = load_model(model_path, tokenizer)