from datasets import Dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_dataset(dataset: Dataset):
    """Split the dataset into training and validation sets.
    Args:
        dataset: A Hugging Face dataset object.
    Returns:
        Tuple of training and validation datasets."""

    logger.info("Splitting dataset into training and validation sets.")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    validation_dataset = dataset["test"]
    train_dataset = dataset["train"]
    logger.info(
        "Dataset split complete. Training samples: %d, Validation samples: %d",
        len(train_dataset),
        len(validation_dataset),
    )

    return train_dataset, validation_dataset
