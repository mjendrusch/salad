import os
import datetime
from typing import Dict

import random
import numpy as np
from numpy import ndarray

from torch.utils.data import Dataset, IterableDataset

class Synthete:
    def __init__(self, path, mode="Train", filter=True,
                 seed=42, p_valid=0.1, p_test=0.1,
                 relative_path="data/npz") -> None:
        super().__init__()
        self.data = np.load(f"{path}/{relative_path}/data.npz")
        self.esm = np.load(f"{path}/{relative_path}/synthete_esm.npz")
        is_good = (self.esm["sc_rmsd"] <= 2.0) * (self.esm["plddt"] > 70) > 0
        self.is_good = is_good.reshape(-1, 11)
        self.is_good_any = self.is_good.any(axis=-1)
        self.filter = filter
        subset = slice(None)
        if self.filter:
            subset = self.is_good_any
        self.seqs = self.esm["seqs"][subset]
        self.ncacocb = self.data["ncacocb"][subset]
        self.mask = self.data["mask"][subset]
        mask = self.mask
        size = mask.shape[0]
        rng = np.random.RandomState(seed)
        index = rng.permutation(size)
        valid_count = int(p_valid * size)
        test_count = int(p_test * size)
        self.index = dict(
            Train=index[:-(valid_count + test_count)],
            Valid=index[-(valid_count + test_count):-test_count],
            Test=index[-test_count:],
        )[mode]

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        idx = self.index[index]
        item = dict(
            ncacocb=self.ncacocb[idx],
            seq=self.seqs[idx].astype(np.int32),
            mask=self.mask[idx])
        mask = item["mask"]
        # sample a random sequence from the set of SALAD & ProteinMPNN sequences
        if self.filter:
            is_good = self.is_good[idx]
            if not is_good.any():
                return None
            aa = item["seq"][is_good.nonzero()[0]]
            aa = aa[np.random.randint(0, aa.shape[0])]
        else:
            aa = item["seq"][np.random.randint(0, 11)]
        # convert positions to masked atom24 format
        atom_pos = np.concatenate((
            item["ncacocb"], np.zeros((256, 19, 3), dtype=np.float32)), axis=1)
        atom_mask = np.zeros((256, 24), dtype=np.bool_)
        atom_mask[:, :5] = True
        atom_mask *= mask[:, None]
        atom_mask = atom_mask > 0
        # build standard residue and chain index, for a single-chain protein
        residue_index = np.arange(256, dtype=np.int32)
        chain_index = np.zeros((256,), dtype=np.int32)
        # return a dictionary compatible with the AllPDB output format
        result = dict(
            mask=mask,
            aa_gt=aa,
            all_atom_positions=atom_pos,
            all_atom_mask=atom_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            batch_index=chain_index
        )
        result["seq_mask"] = result["mask"] * (result["aa_gt"] != 20)
        result["residue_mask"] = result["mask"] * result["all_atom_mask"].any(axis=-1)
        return result

    def __len__(self):
        return len(self.index)

# all items have size 256, if you need more, increase the rebatch value
class SyntheteStream(IterableDataset):
    def __init__(self, path, mode="Train", filter=True, seed=42, p_valid=0.1, p_test=0.1) -> None:
        super().__init__()
        self.synthete = Synthete(path, mode, filter, seed, p_valid, p_test)
        self.current_index = []

    def next_item(self):
        data = None
        while data is None:
            if not self.current_index:
                self.current_index = list(range(len(self.synthete)))
                random.shuffle(self.current_index)
            index = self.current_index.pop()
            data = self.synthete[index]
        return data

    def __iter__(self):
        while True:
            yield self.next_item()
