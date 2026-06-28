from src.finetuning.prepare import prepare_tokenizer,convert_alpaca_format,split_dataset
from src.paths import TOKENIZER_DIR,CHECKPOINT_DIR
from src.finetuning.hf_compatible_model import BetterGPT,BetterGPTConfig

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedTokenizerFast, AutoConfig

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "you are a helpful assistant."},
    {"role": "user", "content": prompt},
    {"role":"assistant", "content": "LLMs are otherwise known as foundation models..."},
    {"role":"user", "content": "What are they good at?"},
    {"role":"assistant","content":"it is based on training data"}
]


if __name__ == "__main__":
    import safetensors.torch as st
    import torch

    path = CHECKPOINT_DIR / "fine-tuned-model"
    tok_path = "./tokenizer_checkpoint/updated-tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained(path)
    msgs = [{"role": "user", "content": "Name a popular author from the 21st Century."}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    

    AutoConfig.register("better_gpt", BetterGPTConfig)
    AutoModelForCausalLM.register(BetterGPTConfig, BetterGPT)
    m1 = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=False)

    assert m1.lm_head.weight.data_ptr() == m1.model.emb_layer.weight.data_ptr()
    m1.eval()
    out1 = m1.custom_generate(ids, max_tokens=50, temp=0.7,top_k=50, eos_id=tokenizer.eos_token_id)
    print(tokenizer.decode(out1[0], skip_special_tokens=False))
    
    m2 = BetterGPT(BetterGPTConfig.from_pretrained(path))
    sd = st.load_file(path / "model.safetensors")
    result = m2.load_state_dict(sd, strict=False)
    print(tokenizer.decode(out1[0], skip_special_tokens=False))
    assert m2.lm_head.weight.data_ptr() == m2.model.emb_layer.weight.data_ptr()
    
    m2.eval()
    
    out2 = m2.custom_generate(ids, max_tokens=50, temp=0.7,top_k=50, eos_id=tokenizer.eos_token_id)
    
    print(tokenizer.decode(out2[0], skip_special_tokens=False))

    # out = m1.generate(
    #     ids,
    #     max_new_tokens=100,
    #     do_sample=True, temperature=0.7, top_k=50,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.pad_token_id or tok.eos_token_id,
    # )

    # print(out)


    

