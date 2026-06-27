from src.finetuning.prepare import prepare_tokenizer
from transformers import PreTrainedTokenizerFast
from src.paths import TOKENIZER_DIR


prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "you are a helpful assistant."},
    {"role": "user", "content": prompt},
    {"role":"assistant", "content": "LLMs are otherwise known as foundation models..."},
    {"role":"user", "content": "What are they good at?"},
    {"role":"assistant","content":"it is based on training data"}
]


if __name__ == "__main__":
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    tokenizer = prepare_tokenizer(tokenizer)
    print(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))