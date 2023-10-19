from typing import Iterator
import torch
from torch.utils.data import RandomSampler


class RandomPairSampler(RandomSampler):
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.arange(0, n, dtype=torch.long).view(-1, 2)[torch.randperm(n // 2, generator=generator)].view(-1).tolist()
            yield from torch.arange(0, n, dtype=torch.long).view(-1, 2)[torch.randperm(n // 2, generator=generator)].view(-1).tolist()[:self.num_samples % n]
