import time
from copy import copy
import random
import numpy as np
from pydssp import assign
from torch.utils.data import IterableDataset, get_worker_info

import salad.data.allpdb as pdb
from salad.data.elements import np_assign_dssp

def _center_fill(pos, mask, residue_type: np.ndarray):
    if residue_type.dtype not in (np.int32, np.int64):
        residue_type = np.argmax(
            residue_type[:, None] == np.array(pdb.RESTYPES), axis=1)
    center_index = pdb.RESTYPE_CENTER[residue_type]
    index = np.arange(center_index.shape[0], dtype=np.int32)
    centers = pos[index, center_index]
    pos = np.where(mask[..., None], pos, centers[:, None])
    return pos

def _select_center(pos, residue_type: np.ndarray):
    numerical_residue_type = np.argmax(residue_type[..., None] == np.array(pdb.RESTYPES), axis=-1)
    center_index = pdb.RESTYPE_CENTER[numerical_residue_type]
    index = np.arange(center_index.shape[0], dtype=np.int32)
    return pos[index, center_index]

def _select_side(pos, residue_type: np.ndarray, residue_name: np.ndarray, atom_name: np.ndarray):
    ca = pos[:, 1]
    cb = pos[:, 4]
    atom = pos[:, 0]
    select_ca = (residue_type == "AA") * (residue_name == "GLY")
    select_cb = (residue_type == "AA") * (residue_name != "GLY")
    select_purine = (residue_name[:, None] == np.array(["A", "G", "DA", "DG"])).any(axis=-1)
    select_pyrimidine = (residue_name[:, None] == np.array(["U", "C", "DT", "DC"])).any(axis=-1)
    c2 = pos[(atom_name == "C2") * select_pyrimidine[:, None]]
    c4 = pos[(atom_name == "C4") * select_purine[:, None]]
    choice_index = np.zeros((atom.shape[0],), dtype=np.int32)
    choice = atom
    choice = np.where(select_ca[..., None], ca, choice)
    choice_index = np.where(select_ca, 1, choice_index)
    choice = np.where(select_cb[..., None], cb, choice)
    choice_index = np.where(select_cb, 4, choice_index)
    if select_purine.any():
        choice[select_purine] = c4
        choice_index = np.where(
            select_purine,
            np.argmax(atom_name == "C4", axis=-1),
            choice_index)
    if select_pyrimidine.any():
        choice[select_pyrimidine] = c2
        choice_index = np.where(
            select_pyrimidine,
            np.argmax(atom_name == "C2", axis=-1),
            choice_index)
    return choice, choice_index

residue_order = pdb.RESIDUE_ORDER

class DropMasked:
    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        if "atom_mask" in item:
            selector = item["atom_mask"].any(axis=1)
        elif "all_atom_mask" in item:
            selector = item["all_atom_mask"].any(axis=1)
        return pdb.slice_dict(item, selector), selected_chain

class DropResidueType:
    def __init__(self, *drop_types: str):
        self.drop_types = drop_types

    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        selector = (item["residue_type"][:, None] != np.array(self.drop_types)).all(axis=-1)
        return pdb.slice_dict(item, selector), selected_chain

class ResidueAdapter:
    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        aa_gt = np.argmax(
            item["residue_name"][:, None] == residue_order[None, :], axis=-1)
        aa_gt = np.where(
            (item["residue_name"][:, None] != residue_order[None, :]).all(axis=-1),
            20,
            aa_gt
        )
        item["aa_gt"] = aa_gt
        item["all_atom_positions"] = item["position"]
        item["all_atom_mask"] = item["atom_mask"]
        return item, selected_chain

class NumericAdapter:
    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        item["chain_index"] = pdb.numerical_chain_index(item["chain_index"])
        item["entity_index"] = pdb.numerical_chain_index(item["entity_index"])
        item["residue_type"] = pdb.numerical_residue_type(item["residue_type"])
        item["atom_type"] = pdb.numerical_atom_type(item["atom_type"][:, 0])
        item["aa_gt"] = np.where(item["residue_type"] < 3, item["aa_gt"], 21 + 5 + 5 + item["atom_type"])
        unk_type = np.full_like(item["aa_gt"], 20)
        unk_type[item["residue_type"] == 1] = 25
        unk_type[item["residue_type"] == 2] = 30
        unk_type[item["residue_type"] > 2] = 30 + 119
        item["unk_type"] = unk_type
        item["aa"] = np.where(item["is_condition"], item["aa_gt"], unk_type)
        # drop atom name & residue name
        del item["atom_name"]
        del item["residue_name"]
        return item, selected_chain

class DropShort:
    def __init__(self, min_size: int = 10):
        self.min_size = min_size

    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        size = (item["chain_index"] == selected_chain).sum()
        if size < self.min_size:
            # print("continue item too short")
            return None, selected_chain
        return item, selected_chain

class DropLong:
    def __init__(self, max_entry_size: int = 2048):
        self.max_entry_size = max_entry_size

    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        size = item["chain_index"].shape[0]
        if size > self.max_entry_size:
            # random contiguous crop at input
            start = np.random.randint(0, size - self.max_entry_size)
            stop = start + self.max_entry_size
            item = pdb.slice_dict(item, slice(start, stop))
            # print("continue item too short")
            return item, selected_chain
        return item, selected_chain

class LigandContext:
    def __init__(self, context_radius: float = 12.0,
                 center_context_radius: float = 22.0,
                 p_joint_design: float = 0.5,
                 p_atomize_aa: float = 0.1,
                 p_condition: float = 0.5,
                 p_drop_atom: float = 0.1,
                 max_atomized: int = 512,
                 max_entry_size: int = 4096,
                 max_chain_size: int = 4096):
        self.context_radius = context_radius
        self.center_context_radius = center_context_radius
        self.p_joint_design = p_joint_design
        self.p_atomize_aa = p_atomize_aa
        self.p_condition = p_condition
        self.p_drop_atom = p_drop_atom
        self.max_atomized = max_atomized
        self.max_entry_size = max_entry_size
        self.max_chain_size = max_chain_size

    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        # massive entry protection
        if item["residue_index"].shape[0] > self.max_entry_size:
            # print("cutting down to size")
            item = pdb.slice_dict(item, item["chain_index"] == selected_chain)
            if item["residue_index"].shape[0] > self.max_chain_size:
                # print("still too large")
                return None, selected_chain
        if random.random() < self.p_joint_design:
            selector = item["residue_type"] == "AA"
        else:
            selector = (
                (item["chain_index"] == selected_chain)
              * (item["residue_type"] == "AA")
            )
        # by default, nothing is an atom
        item["is_atom"] = np.zeros_like(item["residue_index"], dtype=np.bool_)
        # split into amino acid base chains and other molecules
        base_chain = pdb.slice_dict(item, selector)
        other_chains = pdb.slice_dict(item, ~selector)
        base_chain["is_condition"] = np.zeros_like(base_chain["residue_index"], dtype=np.bool_)
        base_pos = base_chain["all_atom_positions"]
        base_mask = base_chain["all_atom_mask"]
        base_pos = _center_fill(base_pos, base_mask, base_chain["residue_type"])
        base_chain["all_atom_positions"] = base_pos
        if other_chains["residue_index"].shape[0] == 0:
            token_count = base_chain["residue_index"].shape[0]
            return base_chain, token_count
        other_pos = other_chains["all_atom_positions"]
        other_mask = other_chains["all_atom_mask"]
        other_pos = _center_fill(other_pos, other_mask, other_chains["residue_type"])
        other_chains["all_atom_positionss"] = other_pos
        # get distance between centers atoms
        base_center = base_pos[:, 1]
        other_center = other_pos[:, 1]
        distance = np.linalg.norm(base_center[:, None] - other_center[None, :], axis=-1)
        other_allowed = (distance < self.center_context_radius).any(axis=0)
        other_chains = pdb.slice_dict(other_chains, other_allowed)
        other_chains["is_condition"] = np.ones_like(other_chains["residue_index"], dtype=np.bool_)
        # split other into NA and AA
        other_aa = pdb.slice_dict(
            other_chains,
            other_chains["residue_type"] == "AA")
        other_x = pdb.slice_dict(
            other_chains,
            other_chains["residue_type"] != "AA")
        # atomize other_x
        atoms = pdb.atomize(other_x, num_atoms=24)
        aa_atoms = pdb.atomize(other_aa, num_atoms=24)
        if random.random() < self.p_atomize_aa:
            atoms = pdb.concatenate_dict((atoms, aa_atoms))
            other_aa = None
        atoms["is_atom"] = np.ones_like(atoms["residue_index"], dtype=np.bool_)
        # cut down to maximum number of atomized
        atom_pos = atoms["all_atom_positions"][:, 0]
        residue_pos = base_pos[base_mask]
        if residue_pos.shape[0] == 0:
            return None, selected_chain
        if atom_pos.shape[0] > 0:
            # do CA distances, everything else is slow
            distance = np.linalg.norm(residue_pos[:, None, 0] - atom_pos[None, :, 0], axis=-1).min(axis=0)
            closest = np.argsort(distance, axis=0)[:self.max_atomized]
            atoms = pdb.slice_dict(atoms, closest)
        # drop random percentage
        drop_index = np.random.rand(atoms["residue_index"].shape[0]) < self.p_drop_atom
        keep_index = ~drop_index
        atoms = pdb.slice_dict(atoms, keep_index)
        # merge features
        other = []
        if other_aa is not None:
            other.append(other_aa)
        if atoms is not None:
            other.append(atoms)
        if random.random() < self.p_condition:
            result = pdb.concatenate_dict([base_chain] + other)
        else:
            result = base_chain
        return result, selected_chain

class NeighbourAtomize:
    def __init__(self, context_radius: float = 12.0,
                 center_context_radius: float = 22.0,
                 p_atomize: float = 0.1,
                 max_atomized: int = 512,
                 max_entry_size: int = 4096,
                 max_chain_size: int = 4096):
        self.context_radius = context_radius
        self.center_context_radius = center_context_radius
        self.p_atomize = p_atomize
        self.max_atomized = max_atomized
        self.max_entry_size = max_entry_size
        self.max_chain_size = max_chain_size

    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        # massive entry protection
        if item["residue_index"].shape[0] > self.max_entry_size:
            return None, selected_chain

        selector = ~item["is_condition"]

        # by default, nothing is an atom
        item["is_atom"] = np.zeros_like(item["residue_index"], dtype=np.bool_)
        # split into base chains and small molecules
        base_chain = pdb.slice_dict(item, selector)
        other_chains = pdb.slice_dict(item, ~selector)
        # base_chain["is_condition"] = np.zeros_like(base_chain["residue_index"], dtype=np.bool_)
        base_pos = base_chain["all_atom_positions"]
        base_mask = base_chain["all_atom_mask"]
        base_pos = _center_fill(base_pos, base_mask, base_chain["residue_type"])
        base_chain["all_atom_positions"] = base_pos
        if other_chains["residue_index"].shape[0] == 0:
            # token_count = base_chain["residue_index"].shape[0]
            return base_chain, selected_chain
        other_pos = other_chains["all_atom_positions"]
        other_mask = other_chains["all_atom_mask"]
        other_pos = _center_fill(other_pos, other_mask, other_chains["residue_type"])
        other_chains["all_atom_positionss"] = other_pos
        # get distance between centers atoms
        base_center = _select_center(base_pos, base_chain["residue_type"])
        other_center = _select_center(other_pos, other_chains["residue_type"])
        distance = np.linalg.norm(base_center[:, None] - other_center[None, :], axis=-1)
        other_allowed = (distance < self.center_context_radius).any(axis=0)
        other_chains = pdb.slice_dict(other_chains, other_allowed)
        # other_chains["is_condition"] = np.ones_like(other_chains["residue_index"], dtype=np.bool_)
        # split into atomized and non-atomized chains
        chain_indices = np.unique(other_chains["chain_index"])
        should_atomize = np.random.rand(*chain_indices.shape) < self.p_atomize
        should_atomize = chain_indices[should_atomize]
        other_keep = pdb.slice_dict(
            other_chains,
            ~(other_chains["chain_index"][:, None] == should_atomize).any(axis=-1))
        other_atomize = pdb.slice_dict(
            other_chains,
            (other_chains["chain_index"][:, None] == should_atomize).any(axis=-1))
        # atomize other_x
        atoms = pdb.atomize(other_atomize, num_atoms=24)
        atoms["is_atom"] = np.ones_like(atoms["residue_index"], dtype=np.bool_)
        # cut down to maximum number of atomized
        atom_pos = atoms["all_atom_positions"]
        residue_pos = base_center
        if residue_pos.shape[0] == 0:
            return None, selected_chain
        if atom_pos.shape[0] > 0:
            # do CA distances, everything else is slow
            distance = np.linalg.norm(residue_pos[:, None] - atom_pos[None, :, 0], axis=-1).min(axis=0)
            closest = np.argsort(distance, axis=0)[:self.max_atomized]
            atoms = pdb.slice_dict(atoms, closest)
        # merge features
        other = [other_keep]
        if atoms is not None:
            other.append(atoms)
        result = pdb.concatenate_dict([base_chain] + other)
        return result, selected_chain

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item: dict, selected_chain: str):
        for tf in self.transforms:
            if item is None:
                return item, selected_chain
            item, selected_chain = tf(item, selected_chain)
        return item, selected_chain

class PackedStream(IterableDataset):
    def __init__(self, base_data: pdb.AllPDB, transform,
                 min_remaining = 100, min_lookahead = 1000, max_tokens = 4096,
                 split_across_workers = True):
        super().__init__()
        self.base_data = base_data
        self.transform = transform
        self.max_tokens = max_tokens
        self.min_remaining = min_remaining
        self.min_lookahead = min_lookahead
        self.split_across_workers = split_across_workers

    def __iter__(self):
        indices = []
        lookahead = []
        while True:
            remaining = self.max_tokens
            batch = []
            indices = indices + lookahead
            lookahead = []
            start = time.time()
            while remaining > self.min_remaining:
                if not indices:
                    indices = list(range(len(self.base_data.data_index["AA"])))
                    # optionally split list of indices
                    # evenly across DataLoader workers
                    if self.split_across_workers:
                        worker_info = get_worker_info()
                        if worker_info is None:
                            continue
                        worker_id = worker_info.id
                        num_workers = worker_info.num_workers
                        indices = indices[worker_id::num_workers]
                    random.shuffle(indices)
                index = indices.pop()
                item, selected_chain = self.base_data.__getitem__("AA", index)
                item, selected_chain = self.transform(item, selected_chain)
                if item is None:
                    # print("continue item None")
                    continue
                num_tokens = item["residue_index"].shape[0]
                if num_tokens > self.max_tokens:
                    # print("continue too large")
                    continue
                if num_tokens <= remaining:
                    # print("inserting into batch")
                    batch.append(item)
                    remaining -= num_tokens
                elif remaining < self.min_lookahead:
                    # print("finish, remaining too small for lookahead")
                    remaining = 0
                else:
                    # print("continue lookahead")
                    lookahead = lookahead + [index]
            # print(len(batch))
            batch = self.pack_batch(batch)
            stop = time.time()
            # print(f"time per batch: {stop - start:.2f} s")
            yield batch

    def pack_batch(self, items):
        current_batch_index = 0
        for item in items:
            item["batch_index"] = np.full_like(item["residue_index"], current_batch_index, dtype=np.int32)
            item["mask"] = np.ones_like(item["residue_index"], dtype=np.bool_)
            current_batch_index += 1
        batch = pdb.concatenate_dict(items)
        batch = pdb.pad_dict(batch, self.max_tokens)
        return batch

class PaddedStream(IterableDataset):
    def __init__(self, base_data, transform,
                 max_tokens=512,
                 split_across_workers=True,
                 verbose=False):
        self.max_tokens = max_tokens
        self.base_data = base_data
        self.transform = transform
        self.split_across_workers = split_across_workers
        self.verbose = verbose
    
    def __iter__(self):
        indices = []
        while True:
            batch = []
            indices = indices
            start = time.time()
            if not indices:
                indices = list(range(len(self.base_data.data_index["AA"])))
                # optionally split list of indices
                # evenly across DataLoader workers
                if self.split_across_workers:
                    worker_info = get_worker_info()
                    if worker_info is None:
                        continue
                    worker_id = worker_info.id
                    num_workers = worker_info.num_workers
                    indices = indices[worker_id::num_workers]
                random.shuffle(indices)
            index = indices.pop()
            item, selected_chain = self.base_data.__getitem__("AA", index)
            item, selected_chain = self.transform(item, selected_chain)
            if item is None:
                if self.verbose:
                    print("continue item None")
                continue
            batch = pdb.pad_dict(item, self.max_tokens)
            stop = time.time()
            if self.verbose:
                print(f"time per batch: {stop - start:.2f} s")
            yield batch

class DSSP:
    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        residue_type = item["residue_type"]
        pos = item["all_atom_positions"]
        dssp = np_assign_dssp(
            pos[:, :4], np.zeros((pos.shape[0],), dtype=np.int32), 
            item["all_atom_mask"][:, :4].all(axis=1))
        # print(dssp)
        # print("shapes", residue_type.shape, dssp.shape)
        dssp = np.where(residue_type != "AA", 5, dssp)
        item["dssp"] = dssp
        return item, selected_chain

class Contacts:
    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        residue_type = item["residue_type"]
        residue_name = item["residue_name"]
        residue = item["residue_index"]
        chain = item["chain_index"]
        atom_name = item["atom_name"]
        mask = item["all_atom_mask"].any(axis=-1)
        pair_mask = mask[:, None] * mask[None, :]
        pos = item["all_atom_positions"]

        pos, distogram_index = _select_side(pos, residue_type, residue_name, atom_name)
        is_contact = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1) < 10.0
        valid = abs(residue[:, None] - residue[None, :]) >= 9
        valid = valid * (chain[:, None] == chain[None, :]) > 0
        valid *= pair_mask
        other_object = (chain[:, None] != chain[None, :])
        other_object += residue_type[:, None] != residue_type[None, :]
        other_object = (other_object > 0) * pair_mask
        num_contacts = (is_contact * valid).sum(axis=1)
        item["num_contacts"] = num_contacts
        num_contacts = (is_contact * other_object).sum(axis=1)
        item["num_interface"] = num_contacts
        item["distogram_index"] = distogram_index
        item["distogram_pos"] = pos
        return item, selected_chain

class DistogramPositions:
    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        residue_type = item["residue_type"]
        residue_name = item["residue_name"]
        atom_name = item["atom_name"]
        pos = item["all_atom_positions"]

        pos, distogram_index = _select_side(pos, residue_type, residue_name, atom_name)
        item["distogram_pos"] = pos
        item["distogram_index"] = distogram_index
        return item, selected_chain

class RandomCrop:
    def __init__(self, size=384):
        self.size = size

    def __call__(self, item: dict, selected_chain: str):
        if item is None:
            return item, selected_chain
        selector = (
            (~item["is_condition"])
            * (item["residue_type"] != "SMOL")
            * (item["residue_type"] != "METAL")
            * (item["residue_type"] != "HOH")
        )
        # by default, nothing is an atom
        item["is_atom"] = np.zeros_like(item["residue_index"], dtype=np.bool_)
        # split into base chain and everything else
        base = pdb.slice_dict(item, selector)
        other = pdb.slice_dict(item, ~selector)
        # choose crop type
        crop_type = random.choice(["spatial", "interface"])
        all_center = _select_center(item["all_atom_positions"], item["residue_type"])
        base_center = _select_center(base["all_atom_positions"], base["residue_type"])
        other_center = _select_center(other["all_atom_positions"], other["residue_type"])
        if crop_type == "spatial":
            # crop around random residue in selected chain
            center_atom = np.random.randint(base_center.shape[0])
            center_pos = base_center[center_atom]
        elif crop_type == "interface":
            # crop around random interface residue in selected chain
            distance = np.linalg.norm(base_center[:, None] - other_center[None, :], axis=-1)
            interface = (distance < 15.0).any(axis=1)
            center_index = np.argmax(np.random.rand(interface.shape[0]) * interface, axis=0)
            center_pos = base_center[center_index]
        distance = np.linalg.norm(center_pos - all_center, axis=-1)
        closest = np.argsort(distance, axis=0)[:self.size]
        allowed = np.zeros((all_center.shape[0],), dtype=np.bool_)
        allowed[closest] = True
        return pdb.slice_dict(item, allowed), selected_chain

class SelectChains:
    def __init__(self, max_chains=2, min_size=10, p_joint=0.1):
        self.max_chains = max_chains
        self.min_size = min_size
        self.p_joint = p_joint

    def __call__(self, item, selected_chain):
        chains = np.unique(item["chain_index"])
        chain_index = np.arange(chains.shape[0], dtype=np.int32)
        is_polymer = (item["residue_type"][:, None] == np.array(["AA", "DNA", "RNA"])).any(axis=-1)
        chain_lengths = np.array([
            ((item["chain_index"] == c) * is_polymer).sum()
            for c in chains
        ])
        max_size = chain_lengths.sum()
        if max_size < self.min_size:
            return None, selected_chain
        # min_size_chains = chain_lengths > self.min_size
        # if not min_size_chains.any():
        #     return None, selected_chain
        # chains = chains[min_size_chains]
        np.random.shuffle(chain_index)
        num_chains = np.random.randint(1, self.max_chains + 1)
        min_num_chains = np.argmax(np.cumsum(chain_lengths[chain_index], axis=0) > self.min_size) + 1
        if min_num_chains > num_chains:
            num_chains = min_num_chains
        selected_chain = chains[chain_index][0]
        non_condition_chains = chains[chain_index][:num_chains]
        non_condition = (item["chain_index"][:, None] == non_condition_chains).any(axis=1)
        non_condition *= is_polymer
        size = non_condition.sum()
        is_condition = ~non_condition
        if random.random() < self.p_joint:
            is_condition = ~is_polymer
        item["is_condition"] = is_condition
        return item, selected_chain

class DistogrammatonStream(IterableDataset):
    def __init__(self, path, seqres_aa="clusterSeqresAA",
                 min_size=50, start_date="01/01/90", cutoff_date="12/31/21",
                 cutoff_resolution=9.0, center_context_radius=100.0,
                 p_joint=0.5, p_atomize=0.05, max_tokens=512,
                 aa_weight=100, na_weight=0, max_entry_size=2048,
                 split_across_workers=True):
        base_data = pdb.AllPDB(
            path, seqres_aa=seqres_aa,
            cutoff_resolution=cutoff_resolution,
            start_date=start_date, cutoff_date=cutoff_date,
            npz_version="_v3")
        transform = Compose([
            DropResidueType("HOH"),
            DropMasked(),
            DropLong(max_entry_size=max_entry_size),
            SelectChains(min_size=min_size, max_chains=2, p_joint=p_joint),
            ResidueAdapter(),
            DSSP(),# Contacts(),
            NeighbourAtomize(
                p_atomize=p_atomize,
                center_context_radius=center_context_radius),
            RandomCrop(size=max_tokens),
            DistogramPositions(),
            NumericAdapter()
        ])
        self.aa_weight = aa_weight
        self.na_weight = na_weight
        self.max_tokens = max_tokens
        self.base_data = base_data
        self.transform = transform
        self.base_aa_indices = list(range(len(self.base_data.data_index["AA"])))
        self.base_na_indices = list(range(len(self.base_data.data_index["NA"])))
        self.split_across_workers = split_across_workers
    
    def __iter__(self):
        aa_indices = []
        na_indices = []
        while True:
            batch = []
            start = time.time()
            if not aa_indices:
                aa_indices = [i for i in self.base_aa_indices]
                # optionally split list of indices
                # evenly across DataLoader workers
                if self.split_across_workers:
                    worker_info = get_worker_info()
                    if worker_info is not None:
                        worker_id = worker_info.id
                        num_workers = worker_info.num_workers
                        aa_indices = aa_indices[worker_id::num_workers]
                random.shuffle(aa_indices)
            if not na_indices:
                na_indices = [i for i in self.base_na_indices]
                # optionally split list of indices
                # evenly across DataLoader workers
                if self.split_across_workers:
                    worker_info = get_worker_info()
                    if worker_info is not None:
                        worker_id = worker_info.id
                        num_workers = worker_info.num_workers
                        na_indices = na_indices[worker_id::num_workers]
                random.shuffle(na_indices)
            kind = random.choice(self.aa_weight * ["AA"] + self.na_weight * ["NA"])
            indices = {"AA": aa_indices, "NA": na_indices}[kind]
            index = indices.pop()
            item, selected_chain = self.base_data.__getitem__(kind, index)
            item, selected_chain = self.transform(item, selected_chain)
            if item is None:
                # print("continue item None")
                continue
            batch = pdb.pad_dict(item, self.max_tokens)
            stop = time.time()
            # print(f"time per batch: {stop - start:.2f} s")
            yield batch

    def split(self, split, seed: int = 42):
        rng = random.Random(seed)
        base_aa_indices = copy(self.base_aa_indices)
        aa_len = len(base_aa_indices)
        base_na_indices = copy(self.base_na_indices)
        na_len = len(base_na_indices)
        rng.shuffle(base_aa_indices)
        rng.shuffle(base_na_indices)
        result = []
        for f in split:
            subset = copy(self)
            subset.base_aa_indices = base_aa_indices[:int(aa_len * f)]
            base_aa_indices = base_aa_indices[int(aa_len * f):]
            subset.base_na_indices = base_na_indices[:int(na_len * f)]
            base_na_indices = base_na_indices[int(na_len * f):]
            result.append(subset)
        return result

    def pack_batch(self, items):
        current_batch_index = 0
        for item in items:
            item["batch_index"] = np.full_like(item["residue_index"], current_batch_index, dtype=np.int32)
            item["mask"] = np.ones_like(item["residue_index"], dtype=np.bool_)
            current_batch_index += 1
        batch = pdb.concatenate_dict(items)
        batch = pdb.pad_dict(batch, self.max_tokens)
        return batch


class MPNNStream(PackedStream):
    def __init__(self, path, seqres_aa="clusterSeqresAA",
                 min_size=50, start_date="01/01/90", cutoff_date="12/31/21",
                 cutoff_resolution=4.0, context_radius=12.0, center_context_radius=22.0,
                 p_joint_design=0.5, p_atomize_aa=0.1, p_condition=0.5,
                 p_drop_atom=0.1, max_atomized=512,
                 min_remaining=100, min_lookahead=1000, max_tokens=4096,
                 split_across_workers=True):
        base_data = pdb.AllPDB(
            path, seqres_aa=seqres_aa,
            cutoff_resolution=cutoff_resolution,
            start_date=start_date, cutoff_date=cutoff_date,
            npz_version="_v3")
        transform = Compose([
            DropResidueType("HOH"), DropMasked(), DropShort(min_size=min_size),
            ResidueAdapter(),
            LigandContext(context_radius=context_radius,
                          center_context_radius=center_context_radius,
                          p_joint_design=p_joint_design,
                          p_atomize_aa=p_atomize_aa,
                          p_condition=p_condition,
                          p_drop_atom=p_drop_atom,
                          max_atomized=max_atomized,
                          max_entry_size=max_tokens,
                          max_chain_size=max_tokens),
            NumericAdapter()
        ])
        super().__init__(
            base_data, transform, min_remaining,
            min_lookahead, max_tokens,
            split_across_workers)
