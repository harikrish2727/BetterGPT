from transformers import AutoTokenizer


tok = AutoTokenizer.from_pretrained("tokenizer_checkpoint")

messages = [
    {"role": "user", "content": "Hello, how are you?"}
]
test_output = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print("Test Output:\n", test_output)

#remove tokenizer helper


