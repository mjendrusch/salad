"""This module implements data loading for AlphaFoldDB structures.

Not used in the manuscript.
"""

import os
import datetime
from typing import Dict, List, Any

import random
import numpy as np
import pandas as pd
from numpy import ndarray

from torch.utils.data import Dataset, IterableDataset

from salad.data.allpdb import AA_ORDER, slice_dict, pad_dict

class AFDB:
    """AlphaFold DB dataset class."""
    def __init__(self, path, stat_path, filter=None, min_plddt=0.0):
        if filter is None:
            filter = lambda x: True
        stats = pd.read_csv(stat_path, sep=",")
        allowed = stats.apply(filter, axis=1)
        stats = stats[allowed]
        self.stats = stats.reset_index(drop=True)
        self.path = path
        self.min_plddt = min_plddt
        self.aa_order = np.array(AA_ORDER)

    def __getitem__(self, index):
        name = self.stats["name"][index]
        raw_data = np.load(f"{self.path}/npz/AF-{name}-F1-model_v4.npz")
        residue_index = raw_data["residue_index"]
        chain_index = np.zeros_like(raw_data["residue_index"]) # all chains are "A" anyway
        entity_index = raw_data["entity_index"]
        all_atom_positions = raw_data["position"]
        all_atom_mask = raw_data["atom_mask"]
        # per-atom plddt is stored in the bfactor field
        plddt = (raw_data["bfactor"] * all_atom_mask).sum(axis=-1) / np.maximum(1, all_atom_mask.sum(axis=-1))
        aa_gt = np.argmax(
            raw_data["residue_name"][:, None] == self.aa_order[None, :], axis=-1)
        aa_gt = np.where(
            (raw_data["residue_name"][:, None] != self.aa_order[None, :]).all(axis=-1),
            20,
            aa_gt
        )
        result = dict(
            aa_gt=aa_gt,
            # for some reason AlphaFold PDB files have no label_seq?
            residue_index=np.arange(residue_index.shape[0], dtype=np.int32),
            chain_index=chain_index,
            entity_index=entity_index,
            all_atom_positions=all_atom_positions,
            all_atom_mask=all_atom_mask,
            plddt=plddt
        )
        if self.min_plddt > 0.0:
            result = slice_dict(result, result["plddt"] >= self.min_plddt)
        return result

    def __len__(self):
        return len(self.stats["name"])

class AFDBStream(IterableDataset):
    """Infinite stream of random AlphaFold DB structures."""
    def __init__(self, path, stat_path, filter=None,
                 length_weights=False, weight_max=1000,
                 weight_exp=1.0, min_plddt=0.0,
                 size=1024, pessimist_size=250,
                 min_size=16, max_size=1024,
                 order_agnostic=False):
        super().__init__()
        self.data = AFDB(path, stat_path, filter=filter)
        self.min_plddt = min_plddt
        self.size = size
        self.pessimist_size = pessimist_size
        self.min_size = min_size
        self.max_size = max_size
        self.order_agnostic = order_agnostic
        # optionally equalize the amount of structures
        # returned for each length
        self.length_weights = None
        self.length_edges = None
        if length_weights:
            length_edges = np.arange(0, 11) * 100
            length_edges[-1] = 1025
            counts, _ = np.histogram(
                np.array(self.data.stats["length"]), bins=length_edges)
            weights = 1 / np.clip(counts, 1, None)
            the_bin = np.argmax(
                (length_edges[:-1] <= weight_max)
              * (weight_max < length_edges[1:]))
            norm_weight = weights[the_bin]
            weights /= norm_weight
            weights = np.clip(weights, 0, 1)
            weights = weights ** weight_exp

            self.length_weights = weights
            self.length_edges = length_edges

    def __iter__(self):
        current_index = []
        queue = []
        while True:
            # if not current_index:
            #     current_index = list(range(len(self.data)))
            #     random.shuffle(current_index)
            item, queue = self.next_item(current_index, queue)
            yield item

    def next_item(self, current_index: List[int], queue: List[Any]):
        result = []
        current_batch = 0
        remaining = self.size
        new_queue = []
        while True:
            # if we have queued items, unqueue these
            # before getting new items from the item
            # index.
            if queue:
                (index, num_aa, data), *queue = queue
            else:
                if not current_index:
                    current_index += list(range(len(self.data)))
                    random.shuffle(current_index)
                index = current_index.pop()
                data = self.data[index]
                num_aa = data["residue_index"].shape[0]
            # skip structures below a minimum size
            # or above a maximum size
            if num_aa < self.min_size:
                continue
            if self.max_size < num_aa:
                continue
            # accept structures with probability
            # proportional to 1 / count(length)
            # this reduces AFDB-cluster's length
            # bias, which would otherwise result
            # in models seeing mostly structures
            # around 100 - 300 amino acids in length
            if self.length_weights is not None:
                the_bin = np.argmax(
                    (self.length_edges[:-1] <= num_aa)
                  * (num_aa < self.length_edges[1:]))
                weight = self.length_weights[the_bin]
                if random.random() > weight:
                    continue
            if num_aa <= remaining:
                data["batch_index"] = np.full_like(
                    data["residue_index"], current_batch)
                current_batch += 1
                remaining -= num_aa
                result.append(data)
            elif remaining < self.pessimist_size:
                break
            else:
                new_queue.append((index, num_aa, data))
        result = {
            name: np.concatenate([d[name] for d in result], axis=0)
            for name in result[0]
        }
        result = pad_dict(result, self.size)
        result["seq_mask"] = result["mask"] * (result["aa_gt"] != 20)
        result["residue_mask"] = result["mask"] * result["all_atom_mask"].any(axis=-1)
        if self.order_agnostic:
            result["residue_index"] = -np.ones_like(result["residue_index"])
        return result, new_queue

class CroppedAFDBStream(IterableDataset):
    """Infinite stream of cropped AlphaFold DB structures."""
    def __init__(self, path, stat_path, filter=None,
                 length_weights=False, weight_max=1000,
                 weight_exp=1.0, min_plddt=0.0,
                 size=256, min_size=16, spatial_crop_monomer=0.0):
        super().__init__()
        self.data = AFDB(path, stat_path, filter=filter)
        self.min_plddt = min_plddt
        self.size = size
        self.min_size = min_size
        self.spatial_crop_monomer = spatial_crop_monomer

    def __iter__(self):
        current_index = []
        while True:
            yield self.next_item(current_index)

    def next_item(self, current_index: List[int]):
        while True:
            if not current_index:
                current_index += list(range(len(self.data)))
                random.shuffle(current_index)
            index = current_index.pop()
            data = self.data[index]
            data = slice_dict(data, data["aa_gt"] != 20)
            num_aa = data["residue_index"].shape[0]
            # skip structures below a minimum size
            if num_aa < self.min_size:
                continue
            # skip short structures as in AlphaFold2 training
            skip_chance = 1 - max(min(num_aa, 512), 256) / 512
            if random.random() < skip_chance:
                continue
            data = self.crop(data)
            del data["plddt"]
            data["batch_index"] = np.zeros_like(data["chain_index"])
            data["seq_mask"] = data["mask"] * (data["aa_gt"] != 20)
            data["residue_mask"] = data["mask"] * data["all_atom_mask"].any(axis=-1) * (data["aa_gt"] != 20)
            return data

    def crop(self, data):
        chain_index = data["chain_index"]
        length = chain_index.shape[0]
        if length > self.size:
            if random.random() < self.spatial_crop_monomer:
                residue = random.randrange(0, length)
                ca = data["pos"][:, 1]
                residue_ca = ca[residue]
                nearest = np.argsort(np.linalg.norm(residue_ca - ca, axis=-1), axis=0)[:self.size]
                data = slice_dict(data, nearest)
            else:
                start = random.randrange(0, length - self.size)
                end = start + self.size
                data = slice_dict(data, slice(start, end))
        data = pad_dict(data, self.size)
        return data