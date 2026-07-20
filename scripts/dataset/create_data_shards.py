"""
This script is responsible for creating data shards from the dataset.
It uses the ShardDataset class to process the data and create shards for both
training and validation datasets.
The shards are created in the specified data directory,
and the tokenizer is loaded from the specified tokenizer
"""
from datasets import load_dataset,interleave_datasets,Features, Value

from src.utils.paths import DATA_DIR
from src.data_preparation.make_shards import ShardDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

data_path = DATA_DIR

ds_khan = load_dataset(
    "HuggingFaceTB/cosmopedia",
    "khanacademy",
    split="train",
    streaming=True,
)

ds_auto = load_dataset(
    "HuggingFaceTB/cosmopedia",
    "auto_math_text",
    split="train",
    streaming=True,
)

ds_wiki = load_dataset(
    "HuggingFaceTB/cosmopedia",
    "wikihow",
    split="train",
    streaming=True,
)

cosmopedia_dataset = interleave_datasets(
    [ds_khan, ds_auto, ds_wiki],
    stopping_strategy="all_exhausted").shuffle(seed=43,buffer_size=50_000)


def format_example(example):
    example["text"] = (
        f"Question:\n{example['prompt']}\n\n"
        f"Answer:\n{example['text']}"
    )
    return example

features = Features({
    "text": Value("string")
})

cosmopedia_dataset = cosmopedia_dataset.map(format_example,remove_columns=
                            [
                                "prompt",
                                "text_token_length",
                                "seed_data",
                                "format",
                                "audience"
                                ],features=features
                            )


fineweb_ds = load_dataset(
    path="HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True).shuffle(
    seed=42,
    buffer_size=50_000
)

# skipped = fineweb_ds.skip(4_000_000)

ds_finemath = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
star_coder_dataset = load_dataset("bigcode/starcoderdata",data_dir = "python",split="train",streaming=True)


def main():
    """
    Main function to create data shards for training and validation datasets.
    It initializes the ShardDataset class and runs the sharding process for dataset.
    """
    fineweb_shards = ShardDataset(dataset_name="fineweb-edu")
    finemath_shards = ShardDataset(dataset_name="finemath")
    cosmopedia_shards = ShardDataset(dataset_name="cosmopedia")
    starcoder_shards = ShardDataset(dataset_name="starcoder_python",data_column_name="content")
    
    logger.info("starting data sharding...")

    fineweb_shards.run(fineweb_ds,shard_id=1,max_file_limit=40)
    finemath_shards.run(ds_finemath,max_file_limit=20)
    cosmopedia_shards.run(cosmopedia_dataset,max_file_limit=20)
    starcoder_shards.run(star_coder_dataset,shard_id=1,max_file_limit=20)
    logger.info("training shards created.")


if __name__ == "__main__":
    main()
