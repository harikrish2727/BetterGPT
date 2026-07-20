"""
This script saves the pretrained model saved as .pt file to model.safetensors in the given path.
"""


if __name__ == "__main__":
        
    import torch
    from safetensors.torch import load_file

    from configs.configuration_bettergpt import BetterGPTConfig
    from src.models.modeling_bettergpt import BetterGPT
    from src.models.base_model import BetterGPTModel
    from src.utils.paths import HUB_DIR



    ckpt_path = "./checkpoints/checkpoint_final.pt"   #wherever final pretrained model is, which is a .pt file for my case
    out_dir   = HUB_DIR                        #to huggingface safetensors format

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"error loading checkpoint {e}")

    config = BetterGPTConfig(**ckpt["model_config"])
    model = BetterGPT(config)

    BetterGPTConfig.register_for_auto_class()
    BetterGPTModel.register_for_auto_class("AutoModel")
    BetterGPT.register_for_auto_class("AutoModelForCausalLM")

    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()}   #removing compiled key name changes

    
    model.load_state_dict(state, strict=True)
    print(f"failed to load state dict {e}")

    # refuse to save an untrained model
    g = model.model.rmsnorm.gamma
    assert not torch.equal(g, torch.ones_like(g)), "gamma still at init — checkpoint not applied!"

    
    model.save_pretrained(out_dir)
    print(f"model saving in safetensors format failed {e}")


    sd = load_file(f"{out_dir}/model.safetensors")
    for k, v in sd.items():
        assert torch.equal(v, state[k]), f"file != checkpoint at {k}"
    print("verified: file matches checkpoint,", len(sd), "tensors")