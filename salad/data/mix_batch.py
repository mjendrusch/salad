import random
from torch.utils.data import IterableDataset

class MixBatch(IterableDataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        weights = [w for w, _ in self.datasets]
        iters = [iter(s) for _, s in self.datasets]
        while True:
            yield next(random.choices(iters, weights=weights, k=1)[0])
