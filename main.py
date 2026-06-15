import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import ModelConfig, TrainingConfig
from model import BetterGPT


def load_and_predict(text, model_path, tokenizer_path, device,
                     max_tokens=100, temp=0.4, top_k=7, stop_at_eos=True):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>") if stop_at_eos else None

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = BetterGPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)          

    if isinstance(text, str):
        text = [text]
    enc_batch = tokenizer.encode_batch(text)

    seqs = []
    for enc in enc_batch:
        ids = enc.ids
        if eos_id is not None and ids and ids[-1] == eos_id:
            ids = ids[:-1]            #drop eos token id 
        seqs.append(ids)

    lengths = {len(s) for s in seqs}
    assert len(lengths) == 1, (
        f"prompts differ in length {lengths}; batched gen needs a key-padding "
        f"mask the model lacks. Pass equal-length prompts, or generate one at a time."
    )

    idx = torch.tensor(seqs, device=device)       
    out = model.generate(idx, max_tokens=max_tokens, temp=temp,
                         top_k=top_k, eos_id=eos_id)
    return tokenizer.decode_batch(out.tolist(), skip_special_tokens=True)



if __name__ == "__main__":
    import argparse
    import os

    path = TrainingConfig().checkpoint_dir
    model_path = os.path.join(path, "best_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    
    tokenizer_path = "./tokenizer.json"  
    device = TrainingConfig().device
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=7)
    parser.add_argument("--stop_at_eos", type=bool, default=True)

    args = parser.parse_args()



    output = load_and_predict(
        text=args.text,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_k=args.top_k,
        stop_at_eos=args.stop_at_eos
    )
    print(output)