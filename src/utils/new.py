import torch

from src.models.model import BetterGPT
from configs.model import BetterGPTConfig as ModelConfig
from src.utils.logger import get_logger
from src.utils.paths import CHECKPOINT_DIR,TOKENIZER_DIR


logger = get_logger(__name__)

def load_model(saved_model_path, tokenizer):
    model = BetterGPT(ModelConfig())
    ckpt = torch.load(
        saved_model_path, map_location="cpu", weights_only=True
    )
    
    old_state_dict = ckpt["model_state"]

    # new_state_dict = {}
    # for key, value in old_state_dict.items():
    #     if "inv_freq" in key:
    #         continue
    #     if key.startswith("lm_head"):
    #         new_state_dict[key] = value  # lm_head stays top-level
    #     else:
    #         new_state_dict["model." + key] = value

    model.load_state_dict(old_state_dict, strict=False)

    logger.info("Pretrained weights loaded successfully!")

    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    logger.info("Resized token embeddings to match tokenizer size.")
    assert model.lm_head.weight.data_ptr() == model.model.emb_layer.weight.data_ptr(), (
        "tie broke after resize"
    )
    logger.info(f"vocab: {model.lm_head.weight.shape[0]} == {len(tokenizer)}")
    return model


if __name__ == "__main__":
    tokenizer_path = TOKENIZER_DIR
    model_path = "model_checkpoints/best_model.pt"
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    model = load_model(model_path, tokenizer)
