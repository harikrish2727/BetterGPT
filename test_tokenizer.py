from transformers import AutoTokenizer


tok = AutoTokenizer.from_pretrained("tokenizer_checkpoint")



def apply_template(messages):
    messages = [
    {"role": "user", "content": "Hello, how are you?"}
]
    test_output = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print("Test Output:\n", test_output)


def number_test(num_list:list):
    for p in num_list:
        enc = tok(p)
        print(p, "→", [tok.decode(i) for i in enc["input_ids"]])

if __name__ == "__main__":
    nums = ["3.14159", "2026-07-07", " 1234567", "175B","hello"," hello"]
    number_test(nums)