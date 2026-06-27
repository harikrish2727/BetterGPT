from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]



CHECKPOINT_DIR = ROOT_DIR / "model_checkpoints"

TOKENIZER_DIR = ROOT_DIR / "tokenizer_checkpoint"

DATA_DIR = ROOT_DIR / "data/TinyStories_data"
