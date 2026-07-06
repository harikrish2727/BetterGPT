from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]  # change root 2


CHECKPOINT_DIR = ROOT_DIR / "model_checkpoints"

TOKENIZER_DIR = ROOT_DIR / "tokenizer_checkpoint"

DATA_DIR = ROOT_DIR / "data"

FINETUNE_OUT_DIR = ROOT_DIR / "checkpoints" / "finetuned_model_files"

LOG_DIR = ROOT_DIR / "logs"

