import torch

from src.finetuning.template import chat_template
from src.finetuning.hf_compatible_model import BetterGPTConfig, BetterGPT
from configs.ft_config import sft_config



def split_dataset(dataset):
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    validation_dataset = dataset["test"]
    train_dataset = dataset["train"]
    return train_dataset, validation_dataset



def prepare_tokenizer(tokenizer):
    additional_special_tokens = ["<|system|>","<|user|>","<|assistant|>","<|im_start|>","<|im_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens":additional_special_tokens})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = chat_template
    return tokenizer


def convert_alpaca_format(examples):
    conversations = []
    for inst, inp, out in zip(
        examples["instruction"],
        examples.get("input", [""]*len(examples["instruction"])),
        examples["output"]
    ):

        user_content = inst
        if inp and inp.strip():
            user_content = f"{inst}\n\n{inp}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": out}
        ]
        conversations.append(messages)
    return {"messages": conversations}



def load_model(saved_model_path,tokenizer):
    """changed pretrained model class after training, this function will safely load the trained model for fien tuning,
    also we can safely change model seq len"""
    model = BetterGPT(BetterGPTConfig())
    ckpt = torch.load(saved_model_path/"model.pt", map_location="cpu", weights_only=True)  #sft trainer needs cpu, it will load to gpu later
    old_state_dict = ckpt["model_state"]

    new_state_dict = {}

    for key, value in old_state_dict.items():
    #Skip loading RoPE buffers to adjust seq leng later
        if "rope.sin" in key or "rope.cos" in key:
            continue
        
        if key.startswith("lm_head"):
            new_state_dict[key] = value
        else:
            new_state_dict["model." + key] = value

    #strict=False as skipped rope keys
    model.load_state_dict(new_state_dict, strict=False)
    print("Pretrained weights loaded successfully!")
    model.resize_token_embeddings(len(tokenizer),mean_resizing=False)
    print("Resized token embeddings to match tokenizer size.")
    return model



