"""
Fine-tuning script for the pretrained BetterGPT model on the Alpaca dataset.

"""
from trl import SFTTrainer
from transformers import PreTrainedTokenizerFast,EarlyStoppingCallback
from datasets import load_dataset

from configs.ft_config import sft_config
from src.utils.model_loader import load_model
from src.utils.ft_tokenizer_helper import prepare_tokenizer
from src.utils.prepare_alpaca_dataset import convert_alpaca_format
from src.utils.paths import CHECKPOINT_DIR, TOKENIZER_DIR
from src.utils.data_splitter import split_dataset
from src.utils.logger import get_logger


logger = get_logger(__name__)

model_path = CHECKPOINT_DIR
tokenizer_path = TOKENIZER_DIR


def main():
    """
    Main function to orchestrate the fine-tuning process.
    """
    try:
        logger.info("Starting fine-tuning process")
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        logger.info("Alpaca Dataset loaded successfully")
    except Exception as e:
        logger.error(f"Error occurred while loading dataset or tokenizer: {e}")
        raise
    try:
        dataset = dataset.map(convert_alpaca_format, batched=True)
        logger.info("Alpaca Dataset converted to chat format successfully")
    except Exception as e:
        logger.error(f"Error occurred while converting dataset to chat format: {e}")
        raise
    try:
        train_dataset, eval_dataset = split_dataset(dataset)
        logger.info("Alpaca Dataset split into training and validation sets successfully")
    except Exception as e:
        logger.error(f"Error occurred while splitting dataset: {e}")
        raise
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error occurred while loading tokenizer: {e}")
        raise
    try:
        tokenizer = prepare_tokenizer(tokenizer)
        logger.info("Tokenizer prepared successfully")
    except Exception as e:
        logger.error(f"Error occurred while preparing tokenizer: {e}")
        raise
    try:
        model = load_model(model_path, tokenizer)
        logger.info("Model retied and loaded successfully")
    except Exception as e:
        logger.error(f"Error occurred while loading model: {e}")
        raise
    return model, train_dataset, eval_dataset, tokenizer


if __name__ == "__main__":
    model, train_dataset, eval_dataset, tokenizer = main()

    model = load_model(model_path,tokenizer)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{num_params/1e6:.2f}M parameters")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    logger.info("Fine-tuning completed successfully. Saving model and tokenizer...")

    # trainer.save_model()
    # tokenizer.save_pretrained(tokenizer_path)
    trainer.save_model()
