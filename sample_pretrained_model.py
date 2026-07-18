"""
This script loads the pretrained model and tokenizer saved in huggingface expected format aand give sample predictions.
Make sure the below mentioned files and tokenizer files from tokenizer_checkpoint directory, are in the model file paths to work this "AutoTokenizer.from_pretrained(path,trust_remote_code=True)",

attention.py,base_model.py,model.py,model_config.py,layer_normalization.py,positional_embeddings.py,rope_helper.py,swiglu_feed_forward.py,trasformer_block.py and logger.py.
These files are needed for the model to make generation work for this project.
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(path,trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"error loading tokenizer {e}")

def tokenize(text,tokenizer):
    return torch.tensor((tokenizer(text)["input_ids"][:-1]),device=device).reshape(1,-1)
        

def load_model(path):
    try:
        model = AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True)
        model = model.to(device)
        return model
    except Exception as e:
        print("failed loading model")
        return (e)



def predict(ids,model,tokenizer,max_new_tokens=100):
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0],skip_special_tokens=True)



if __name__ =="__main__":
    from src.utils.paths import CHECKPOINT_DIR,TOKENIZER_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = CHECKPOINT_DIR

    model = load_model(path)
    tokenizer = load_tokenizer(path)

    text = "The "

    ids = tokenize(text,tokenizer)

    model_response = predict(ids,model,tokenizer,max_new_tokens=100)

    print(f"model response:  {model_response}")
