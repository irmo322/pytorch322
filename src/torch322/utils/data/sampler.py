from torch.utils.data import Sampler
import random


class ChunkSampler(Sampler[int]):

    def __init__(self, dataset_size: int, chunk_size: int, shuffle: bool = False):
        """
        A sampler that yields exactly `chunk_size` indices per iteration,
        maintaining a persistent cursor across loops.
        When the cursor reaches the end of the dataset, it wraps around to the
        beginning (and reshuffles the indices if shuffle=True).
        """
        super().__init__()
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be positive, got {dataset_size}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.dataset_size = dataset_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle

        self._cursor = 0
        self._indices = list(range(self.dataset_size))
        self.reset()

    def reset(self):
        """Reset the cursor and optionally reshuffle the indices."""
        self._cursor = 0
        if self.shuffle:
            random.shuffle(self._indices)

    def __iter__(self):
        for _ in range(self.chunk_size):
            yield self._indices[self._cursor]
            self._cursor += 1
            if self._cursor == self.dataset_size:
                self.reset()

    def __len__(self):
        return self.chunk_size
