import torch

from trl import SFTTrainer
from transformers import EarlyStoppingCallback
from transformers import PreTrainedTokenizerFast,EarlyStoppingCallback
from datasets import load_dataset

from configs.ft_config import sft_config
from src.utils.model_loader import load_model
from src.utils.ft_tokenizer_helper import prepare_tokenizer
from src.utils.prepare_alpaca_dataset import convert_alpaca_format
from src.utils.paths import CHECKPOINT_DIR, TOKENIZER_DIR
from src.utils.data_splitter import split_dataset

model_path = CHECKPOINT_DIR
tokenizer_path = TOKENIZER_DIR


dataset = load_dataset("yahma/alpaca-cleaned",split="train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


if __name__ == "__main__":
    #load and prepare the dataset
    
    dataset = dataset.map(convert_alpaca_format,remove_columns=dataset.column_names,batched=True)
    print("dataset downloaded")
    train_dataset, eval_dataset = split_dataset(dataset)
    print("dataset prepared")
   #load and prepare the tokenizer
    
    tokenizer = prepare_tokenizer(tokenizer)
    print("tokenizer loaded and applied chat template")

    model = load_model(model_path,tokenizer)
    print("model loaded")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"{num_params/1e6:.2f}M parameters")

    

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    trainer.save_model(model_path/"fine-tuned-model")
    tokenizer.save_pretrained(tokenizer_path/"updated-tokenizer")


    