import os
import numpy as np
from datasets import Dataset,load_dataset
from tokenizers import Tokenizer

class ShardDataset:
  def __init__(
      self,
      dataset_name:str,
      tokenizer: Tokenizer,
      out_dir:str,
      split: str="train",
      data_column_name:str = "text",
      buffer_size:int = 2_0000_000
      ):

    self.tokenizer = tokenizer
    # self.split = split
    # self.data = load_dataset(dataset_name,split=self.split,streaming=True).shuffle(seed=42, buffer_size=10_000)
    self.dataset_name = dataset_name
    data_folder_name = f"{dataset_name.split("/")[-1]}_data"
    self.data_column_name = data_column_name
    self.out_dir = os.path.join(out_dir,data_folder_name)
    self.buffer_size = buffer_size

    self.vocab_size = int(tokenizer.get_vocab_size())
    os.makedirs(self.out_dir,exist_ok = True)

    assert self.vocab_size<65000, "vocab size more, can't store data in uint16"

  def save_shards(self, data_ids, shard_name):
    shard_path = os.path.join(self.out_dir, shard_name)

    if not os.path.exists(shard_path):
        data_ids.tofile(shard_path)
        print(f"{shard_name} saved")


  def run(self,split):
    shard_id = 1
    buffer = np.empty(self.buffer_size, dtype=np.uint16)
    buffer_idx = 0

    ds = load_dataset(self.dataset_name,split=split,streaming=True).shuffle(seed=42, buffer_size=10_000)

    for samples in ds.iter(1000):
      encoded = self.tokenizer.encode_batch(samples[self.data_column_name])

      for item in encoded:
        ids = np.array(item.ids,dtype=np.uint16)
        start=0

        while start<len(ids):

          remaining = self.buffer_size-buffer_idx
          take = min(remaining,len(ids)-start)

          buffer[buffer_idx:buffer_idx + take] = ids[start:start + take]

          buffer_idx += take
          start += take

          if buffer_idx == self.buffer_size:
            shard_name = f"{split}_shard{shard_id:04d}.bin"

            self.save_shards(buffer, shard_name)

            if shard_id % 5 == 0:
              print(f"Created {shard_id} shards")

            shard_id += 1
            buffer_idx = 0

    if buffer_idx>0:
      shard_name = f"{split}_shard{shard_id:04d}.bin"
      self.save_shards(buffer[:buffer_idx], shard_name)


if __name__ == "__main__":
    tok = Tokenizer.from_file("./tokenizer.json")
    shards = ShardDataset(dataset_name="roneneldan/TinyStories",tokenizer=tok,out_dir="./data_shards/")
    # shards.run("validation")
    shards.run("train")