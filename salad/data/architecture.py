import random
import numpy as np
from salad.data.elements import to_elements
from salad.data.allpdb import pad_dict, concatenate_dict

from torch.utils.data import IterableDataset

class Architecture:
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        data = self.base_dataset(index)
        elements = to_elements(data)
        # add index
        elements["element_index"] = np.arange(elements["dssp"].shape[0])
        return data, elements
    
    def __len__(self):
        return len(self.base_dataset)

class ArchitectureStream(IterableDataset):
    def __init__(self, base_dataset, num_centers=1024):
        super().__init__()
        self.data = Architecture(base_dataset)
        self.num_centers = num_centers

    def __iter__(self):
        queue = []
        current = []
        total = 0
        while True:
            if not queue:
                queue = list(range(len(self.data)))
                random.shuffle(queue)
            index, *queue = queue
            _, elements = self.data[index]
            num_elements = elements["dssp"].shape[0]
            if num_elements > self.num_centers:
                continue
            if total + num_elements > self.num_centers:
                yield pad_dict(concatenate_dict(current), self.num_centers)
                current = []
                total = 0
            else:
                elements["batch_index"] = np.array(num_elements * [len(current)], dtype=np.int32)
                current.append(elements)
                total += num_elements
