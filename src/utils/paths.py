from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]  #change root 2



CHECKPOINT_DIR = ROOT_DIR / "model_checkpoints"

TOKENIZER_DIR = ROOT_DIR / "tokenizer_checkpoint"

DATA_DIR = ROOT_DIR / "data"

FINETUNE_OUT_DIR = ROOT_DIR / "checkpoints" / "finetuned_model_files"

LOG_DIR = ROOT_DIR / "logs"

# if __name__ == "__main__":
#     print(f"Root directory: {ROOT_DIR}")
#     print(f"Checkpoint directory: {CHECKPOINT_DIR}")
#     print(f"Tokenizer directory: {TOKENIZER_DIR}")
#     print(f"Data directory: {DATA_DIR}")
#     print(f"Finetune output directory: {FINETUNE_OUT_DIR}")
#     print(f"Log directory: {LOG_DIR}")