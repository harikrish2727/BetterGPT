
import random
import torch

from glob import glob
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info


class TinyDataset(IterableDataset):
    """
    Contiguous-packing dataset for from-scratch LM pretraining (single-GPU).

    Shards are flat uint16 token streams with BOS/EOS already inserted by the
    tokenizer's post-processor. Each shard is cut into non-overlapping
    [seq_length] windows; targets are the same window shifted one token right.
    Windows pass through a shuffle buffer so ordering is randomized without
    random-offset sampling (which would oversample some tokens and skip others).

    Use a num_workers value that divides the shard count so the per-worker file
    split is exact (24 train shards -> 1, 2, 3, 4, 6, 8, 12, 24).
    """

    def __init__(self, path, seq_length, split, shuffle_buffer=4096,
                 infinite=True, seed=1337):
        super().__init__()
        self.data_files = sorted(glob(f"{path}/{split}*.bin"))
        self.seq_length = seq_length
        self.shuffle_buffer = shuffle_buffer
        self.infinite = infinite
        self.seed = seed
        if not self.data_files:
            raise FileNotFoundError(f"No files found for split '{split}' in {path}")

    def _worker_files(self):
        info = get_worker_info()
        if info is None:
            return self.data_files, 0
        files = self.data_files[info.id::info.num_workers]
        return files, info.id

    def _chunks_from_file(self, file, rng):
        try:
            data = np.memmap(file, dtype=np.uint16, mode="r")
            n = len(data)
            n_chunks = (n - 1) // self.seq_length   # -1 because y is shifted +1
            if n_chunks <= 0:
                return
            order = list(range(n_chunks))
            rng.shuffle(order)
            for c in order:
                start = c * self.seq_length
                x = data[start:start + self.seq_length].astype(np.int64)
                y = data[start + 1:start + self.seq_length + 1].astype(np.int64)
                yield torch.from_numpy(x), torch.from_numpy(y)
        finally:
            if hasattr(data, '_mmap') and data._mmap is not None:
                data._mmap.close()

    def _stream(self, files, rng):
        file_order = list(files)
        rng.shuffle(file_order)
        for file in file_order:
            yield from self._chunks_from_file(file, rng)

    def __iter__(self):
        files, worker_id = self._worker_files()
        if not files:
            return
        rng = random.Random(self.seed + worker_id)

        buf = []
        epoch = 0
        while True:
            for sample in self._stream(files, rng):
                if len(buf) < self.shuffle_buffer:
                    buf.append(sample)
                    continue
                j = rng.randrange(len(buf))
                out, buf[j] = buf[j], sample
                yield out
            if not self.infinite:
                break
            epoch += 1
            rng = random.Random(self.seed + worker_id + epoch * 9973)

        rng.shuffle(buf)
        for sample in buf:
            yield sample
