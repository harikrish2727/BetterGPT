from src.paths import DATA_DIR
from tokenizers import Tokenizer
from src.data_preparation.make_shards import ShardDataset


tokenizer_path = "./tokenizer_checkpoint/tokenizer.json"
data_path = DATA_DIR/"new"

if __name__ == "__main__":
    tok = Tokenizer.from_file(tokenizer_path)
    shards = ShardDataset(dataset_name="roneneldan/TinyStories",tokenizer=tok,out_dir=data_path)
    shards.run("validation")
    shards.run("train")