from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from src.paths import DATA_DIR,TOKENIZER_DIR
from src.data_preparation.make_shards import ShardDataset
from src.logger import get_logger


logger = get_logger(__name__)


tokenizer_path = TOKENIZER_DIR/"updated-tokenizer"
data_path = DATA_DIR

if __name__ == "__main__":

    shards = ShardDataset()
    logger.info("starting data sharding...")
    shards.run("validation")
    logger.info("validation shards created.")
    shards.run("train")
    logger.info("training shards created.")
