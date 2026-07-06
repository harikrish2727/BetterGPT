"""
This script is responsible for creating data shards from the dataset. 
It uses the ShardDataset class to process the data and create shards for both 
training and validation datasets.
The shards are created in the specified data directory, 
and the tokenizer is loaded from the specified tokenizer
"""

from src.utils.paths import DATA_DIR
from src.data_preparation.make_shards import ShardDataset
from src.utils.logger import get_logger


logger = get_logger(__name__)


data_path = DATA_DIR

def main():
    """
    Main function to create data shards for training and validation datasets.
    It initializes the ShardDataset class and runs the sharding process for dataset.
    """
    shards = ShardDataset()
    logger.info("starting data sharding...")
    shards.run("validation")
    logger.info("validation shards created.")
    shards.run("train")
    logger.info("training shards created.")

if __name__ == "__main__":
    main()