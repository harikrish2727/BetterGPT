"""
This script loads the pretrained model and tokenizer saved in huggingface expected format aand give sample predictions.
Make sure the below mentioned files and tokenizer files from tokenizer_checkpoint directory, are in the model file paths to work this "AutoTokenizer.from_pretrained(path,trust_remote_code=True)",

attention.py,base_model.py,model.py,model_config.py,layer_normalization.py,positional_embeddings.py,rope_helper.py,swiglu_feed_forward.py,trasformer_block.py and logger.py.
These files are needed for the model to make generation work for this project.
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



if __name__ =="__main__":

    from src.utils.paths import HUB_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = HUB_DIR

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    model_head = AutoModel.from_pretrained(
        path,
        trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True
    )

    print("tokenizer loaded")

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        device_map="auto"
    )

    print("model loaded")

    ### Generate text

    prompt = "The future of artificial intelligence is"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
