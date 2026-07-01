import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from configs.model import ModelConfig
from configs.training import TrainingConfig
from src.models.model import BetterGPT
from src.logger import get_logger

logger = get_logger("sample")


def load_and_predict(text, model_path, tokenizer_path, device,
                     max_tokens=100, temp=0.4, top_k=7, stop_at_eos=True):
    """Load a trained checkpoint and generate text for one or more prompts.

    All prompts in `text` must encode to the same token length because the model
    uses batched generation without a key-padding mask. For prompts of different
    lengths, call this function once per prompt.

    Args:
        text: A single prompt string or a list of equal-length prompt strings.
        model_path: Path to the .pt checkpoint file produced by the trainer.
        tokenizer_path: Path to the tokenizer JSON file.
        device: Torch device string ('cpu', 'cuda', etc.).
        max_tokens: Maximum number of new tokens to generate per prompt.
        temp: Sampling temperature; 0 for greedy decoding.
        top_k: Restrict sampling to the top-k logits; None for unrestricted.
        stop_at_eos: Stop generation when the <eos> token is produced.

    Returns:
        List of decoded output strings, one per prompt.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
    if temp < 0:
        raise ValueError(f"temp must be >= 0, got {temp}")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>") if stop_at_eos else None

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = BetterGPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    logger.info(
        "Model loaded from %s | vocab_size=%d | emb_dim=%d | num_blocks=%d",
        model_path,
        ckpt["model_config"].get("vocab_size"),
        ckpt["model_config"].get("emb_dim"),
        ckpt["model_config"].get("num_blocks"),
    )

    if isinstance(text, str):
        text = [text]
    enc_batch = tokenizer.encode_batch(text)

    seqs = []
    for enc in enc_batch:
        ids = enc.ids
        if eos_id is not None and ids and ids[-1] == eos_id:
            ids = ids[:-1]    # drop trailing eos token
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

    training_config = TrainingConfig()
    path = training_config.checkpoint_dir
    model_path = os.path.join(path, "best_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    tokenizer_path = "./tokenizer.json"
    device = training_config.device

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--temp", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=7)
    parser.add_argument("--stop_at_eos", action="store_true", default=True)

    args = parser.parse_args()

    output = load_and_predict(
        text=args.text,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_k=args.top_k,
        stop_at_eos=args.stop_at_eos,
    )
    print(output)
