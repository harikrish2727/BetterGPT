from transformers import PreTrainedTokenizerFast, AutoTokenizer
from src.utils.paths import TOKENIZER_DIR
from pathlib import Path


tok = AutoTokenizer.from_pretrained("tokenizer_checkpoint")

# tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)