import torch
from jinja2 import TemplateError

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.modeling_bettergpt import BetterGPT
from src.utils.paths import FINETUNE_OUT_DIR
from configs.configuration_bettergpt import BetterGPTConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

alpaca_path = FINETUNE_OUT_DIR/"alpaca_model"

def prepare_prompt(query:str)->torch.tensor:
    """
    This function applies chat template to the user query and tokenize it.
    """
    msgs = [{"role": "user", "content": query}]
    try:
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    except TemplateError as e:
        logger.debug(f"jinja template error {e}")
    except ValueError as e:
        logger.debug(f"error applying chat template/tokenizing {e}")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    return ids


def generate(ids:torch.tensor):
    """
    this is a test function to check the finetuned model's output.
    """
    try:
        out = model.generate(
            ids,
            max_new_tokens=100,
            do_sample=True, top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False
        )
        
    except ValueError as e:
        logger.debug(f"error in generating output {e}")

    return out


if __name__ == "__main__":

    AutoConfig.register(model_type="better_gpt",config=BetterGPTConfig)  #register config
    AutoModelForCausalLM.register(BetterGPTConfig,BetterGPT)  #register model

    model = AutoModelForCausalLM.from_pretrained(alpaca_path)
    tokenizer = AutoTokenizer.from_pretrained(alpaca_path)

    query = "How did Julius Caesar die?"

    ids = prepare_prompt(query)
    model.eval()
    response = generate(ids)

    print(tokenizer.decode(response[0], skip_special_tokens=True))





