import os
import datetime
from typing import Dict

import random
import numpy as np
from numpy import ndarray

from torch.utils.data import Dataset, IterableDataset
from salad.data.periodic_table import periodic_table, pt_at, ATOM_TYPE_ORDER, index_atoms, apply_pt

# FIXME: diagnostics
def decode_sequence(x: np.ndarray) -> str:
    AA_CODE = "ARNDCQEGHILKMFPSTWYVX"
    x = np.array(x)
    return "".join([AA_CODE[c] for c in x])

# residue types in PDB
RESTYPES = ["AA", "DNA", "RNA", "METAL", "SMOL", "HOH"]
# non-water residue types
RESTYPES_NO_WATER = ["AA", "DNA", "RNA", "METAL", "SMOL"]
# list of amino acid 3-letter codes in alphabetical order
AA_ORDER = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN',
            'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
            'TYR', 'VAL', 'UNK']
# list of amino acid backbone atom names
AA_BACKBONE_ATOMS = ["N", "CA", "C", "O"]
# list of nucleic acid backbone atom names
NA_BACKBONE_ATOMS = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C1'", "C2'", "O2'", "C3'", "O3'"]

class AllPDB:
    def __init__(self, path, start_date="01/01/90", cutoff_date="12/31/21",
                 cutoff_resolution=4.0, filter_residue_type=None,
                 seqres_aa="clusterSeqresAA", seqres_na="clusterSeqresNA",
                 assembly=True, split_types=False, npz_version="") -> None:
        super().__init__()
        self.path = path
        self.split_types = split_types
        compute_data_index = compute_index if assembly else compute_asym_index
        self.data_index = compute_data_index(path, start_date=start_date,
                                             cutoff_date=cutoff_date,
                                             cutoff_resolution=cutoff_resolution,
                                             seqres_aa=seqres_aa,
                                             seqres_na=seqres_na,
                                             npz_version=npz_version)
        self.filter_residue_type = np.array(filter_residue_type or ["AA", "NA", "SMOL"])
        self.fields = [
            "residue_type", "residue_name",
            "residue_index", "chain_index",
            "entity_index", "molecule_index",
            "atom_mask", "position",
            "atom_name", "atom_type"
        ]

    def __getitem__(self, kind, index) -> Dict[str, np.ndarray]:
        data_index = self.data_index[kind]
        chosen_chain = random.choice(data_index[index])
        assembly = random.choice(chosen_chain["assemblies"])
        raw_data = np.load(f"{self.path}/{assembly}")
        residue_type = raw_data["residue_type"]
        if self.split_types:
            split_data = dict()
            for res_type in self.filter_residue_type:
                accept = residue_type == res_type
                raw_data_part = {
                    name: raw_data[name][accept]
                    for name in self.fields
                }
                split_data[res_type] = raw_data_part
            raw_data = split_data
        else:
            accept = (residue_type[:, None] == self.filter_residue_type[None, :]).any(axis=-1)
            raw_data = {
                name: raw_data[name][accept]
                for name in self.fields
            }
        return raw_data, chosen_chain["chain"]

class AllPDBSample(AllPDB):
    def __init__(self, path, start_date="01/01/90", cutoff_date="12/31/21",
                 cutoff_resolution=4, filter_residue_type=None,
                 seqres_aa="clusterSeqresAA", seqres_na="clusterSeqresNA",
                 assembly=True) -> None:
        super().__init__(
            path, start_date, cutoff_date, cutoff_resolution,
            filter_residue_type, seqres_aa, seqres_na, assembly)
        self.data_index = {
            kind: create_cluster_weights(self.data_index[kind])
            for kind in self.data_index
        }


    def __getitem__(self, kind, index) -> Dict[str, np.ndarray]:
        data_index = self.data_index[kind]
        chosen_chain = data_index[index]
        assembly = random.choice(chosen_chain["assemblies"])
        weight = chosen_chain["weight"]
        accept_item = random.random() < weight
        if not accept_item:
            return None
        raw_data = np.load(f"{self.path}/{assembly}")
        residue_type = raw_data["residue_type"]
        accept = (residue_type[:, None] == self.filter_residue_type[None, :]).any(axis=-1)
        raw_data = {
            name: raw_data[name][accept]
            for name in self.fields
        }
        return raw_data, chosen_chain["chain"]

class ProteinPDB(AllPDB):
    def __init__(self, path, start_date="01/01/90",
                 cutoff_date="12/31/21",
                 cutoff_resolution=4,
                 seqres_aa="clusterSeqresAA",
                 seqres_na="clusterSeqresNA",
                 assembly=True) -> None:
        super().__init__(
            path, start_date, cutoff_date,
            cutoff_resolution, ["AA"],
            seqres_aa=seqres_aa,
            seqres_na=seqres_na,
            assembly=assembly)
        self.aa_order = np.array(
            ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN',
             'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
             'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
             'TYR', 'VAL', 'UNK'])

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        raw_data, chain = super().__getitem__("AA", index)
        aa_gt = np.argmax(
            raw_data["residue_name"][:, None] == self.aa_order[None, :], axis=-1)
        aa_gt = np.where(
            (raw_data["residue_name"][:, None] != self.aa_order[None, :]).all(axis=-1),
            20,
            aa_gt
        )
        residue_index = raw_data["residue_index"]
        chain_index = raw_data["chain_index"]
        entity_index = raw_data["entity_index"]
        all_atom_positions = raw_data["position"]
        all_atom_mask = raw_data["atom_mask"]
        return dict(
            aa_gt=aa_gt,
            residue_index=residue_index,
            chain_index=chain_index,
            entity_index=entity_index,
            all_atom_positions=all_atom_positions,
            all_atom_mask=all_atom_mask
        ), chain
    
    def __len__(self) -> int:
        return len(self.data_index["AA"])

class AtomPDB(AllPDB):
    def __init__(self, path, start_date="01/01/90", cutoff_date="12/31/21",
                 cutoff_resolution=4, seqres_aa="clusterSeqresAA",
                 seqres_na="clusterSeqresNA", assembly=True, split_types=False, npz_version=""):
        super().__init__(
            path, start_date, cutoff_date, cutoff_resolution,
            RESTYPES_NO_WATER, seqres_aa, seqres_na,
            assembly, split_types, npz_version)

    def __getitem__(self, index):
        # repeat per residue features across all
        # potential atoms (24 in atom24 format)
        def _irepeat(x):
            return np.repeat(x[:, None], 24, axis=1)
        # get an allpdb-npz assembly
        raw_data, chain = super().__getitem__("AA", index)
        mask = raw_data["atom_mask"]
        position = raw_data["position"][mask]
        atom_type = raw_data["atom_type"][mask]
        atom_name = raw_data["atom_name"][mask]
        atom_order_index = raw_data["atom_order_index"][mask]
        residue_type = _irepeat(raw_data["residue_type"])[mask]
        # mark AA backbone atoms so a model can use this
        # information directly and learn priviledged
        # features for amino acids
        is_aa = residue_type == "AA"
        aa_backbone_one_hot = (atom_name[:, None] == AA_BACKBONE_ATOMS) * is_aa[:, None]
        is_aa_backbone = aa_backbone_one_hot.any(axis=1)
        # mark peptide bond atoms for handling bonds in
        # downstream models
        peptide_n = (atom_name == "N") * is_aa_backbone
        peptide_c = (atom_name == "C") * is_aa_backbone
        # mark DNA/RNA backbone atoms as well
        is_na = (residue_type == "DNA") + (residue_type == "RNA") > 0
        na_backbone_one_hot = (atom_name[:, None] == NA_BACKBONE_ATOMS) * is_na[:, None]
        is_na_backbone = na_backbone_one_hot.any(axis=1)
        # mark 5' and 3' bonding atoms
        na_p = (atom_name == "P") * is_na_backbone
        na_o3 = (atom_name == "O3'") * is_na_backbone
        # set up joint privileged atom info
        privileged_atom = np.concatenate((
            aa_backbone_one_hot, na_backbone_one_hot), axis=-1)
        # encode periodic table information
        atom_type_index = index_atoms(atom_type)
        atom_features = apply_pt(atom_type_index)
        # instead of PDB chain indices we use a precomputed
        # molecule index which is unique per covalently-linked
        # molecule in an assembly
        chain_index = _irepeat(raw_data["molecule_index"])[mask]
        residue_index = _irepeat(raw_data["residue_index"])[mask]
        # get bond index and bond type; inter-residue bonds
        # have to be added at a later stage
        bond_index = raw_data["bond_index"][mask]
        bond_type = raw_data["bond_type"][mask]
        # this information will be used inside models to set up
        # (noisy) equilibrium geometry as a model input.
        # when sampling from a model, that information has to be
        # sourced from an idealised _REFERENCE STRUCTURE_.
        return dict(
            position=position,
            reference_position=position,
            atom_type=atom_type_index,
            atom_features=atom_features,
            is_aa=is_aa,
            is_na=is_na,
            is_bb=is_aa_backbone,
            peptide_n=peptide_n,
            peptide_c=peptide_c,
            na_p=na_p,
            na_o3=na_o3,
            privileged_atom=privileged_atom,
            atom_order_index=atom_order_index,
            chain_index=chain_index,
            residue_index=residue_index,
            bond_index=bond_index,
            bond_type=bond_type
        ), chain

class AtomPDBStream(IterableDataset):
    def __init__(self, path, num_atom=1024, num_residue=None, p_mask_aa=0.0,
                 start_date="01/01/90", cutoff_date="12/31/21",
                 cutoff_resolution=4, seqres_aa="clusterSeqresAA",
                 seqres_na="clusterSeqresNA"):
        super().__init__()
        self.num_atom = num_atom
        self.num_residue = num_residue or num_atom // 4
        self.p_mask_aa = p_mask_aa
        self.data = AtomPDB(path, start_date, cutoff_date, cutoff_resolution,
                            seqres_aa=seqres_aa, seqres_na=seqres_na, assembly=True,
                            split_types=False, npz_version="_v3")
        self.current_index = []
        
    def __iter__(self):
        self.data.data_index["AA"]
        while True:
            yield self.next_item()

    def get_next_pdb(self):
        data = None
        while data is None:
            if not self.current_index:
                print("shuffling...")
                self.current_index = list(range(len(self.data)))
                random.shuffle(self.current_index)
            index = self.current_index.pop()
            data = self.data[index]
        return data, index

    def next_item(self):
        total_atoms = 0
        total_residues = 0
        batch = []
        while total_atoms < self.num_atom and total_residues < self.num_residue:
            data = self.get_next_pdb()
            residue_index = data["residue_index"]
            molecule_index = data["molecule_index"]
            is_aa = data["is_aa"]
            is_bb = data["is_bb"]
            # get unique residue index
            unique_residue_values, unique_residue_index = np.unique(
                np.stack((molecule_index, residue_index), axis=-1),
                axis=0, inverse=True)
            # get unique atom index
            atom_index = np.arange(residue_index.shape[0], dtype=np.int32)
            data["atom_index"] = atom_index
            data["unique_residue_index"] = unique_residue_index
            # subsample aa atoms to backbone only
            if self.p_mask_aa > 0.0:
                masked_aas = np.random.rand(unique_residue_values.shape[0]) < self.p_mask_aa
                masked_atoms = masked_aas[unique_residue_index]
                masked_atoms *= is_aa * (~is_bb)
                data = slice_dict(data, ~(masked_atoms > 0))
            # get size
            num_atoms = residue_index.shape[0]
            num_residues = unique_residue_values.shape[0]

            # see if we have to crop or skip this
            if total_atoms + num_atoms <= self.num_atom and \
               total_residues + num_residues <= self.num_residues:
                data["batch_index"] = len(batch) * np.ones_like(residue_index)
                batch.append(data)
                total_atoms += num_atoms
                total_residues += num_residues
            # decide whether to crop
            else:
                # are we in bad crop territory, i.e. too little space
                # to fit a crop?
                # get the average chain size in this PDB file
                is_polymer = data["is_aa"] + data["is_na"] > 0
                _, molecule_size = np.unique(data["moleclue_index"][is_polymer], return_counts=True)
                mean_size = molecule_size.mean()
                remainder = self.num_atom - total_atoms
                if remainder < mean_size * 0.7:
                    break
                # sample an entity type to use as a center
                center_type = random.choice(np.unique(data["residue_type"]))
                # identify all residues of that type and choose one
                # uniformly at random
                this_type = data["residue_type"] == center_type
                center = np.argmax(np.random.rand() * this_type)
                # compute distance to that entity
                dist = np.linalg.norm(data["position"] - data["position"][center])
                # and select the remainder of residues
                crop_index = np.argsort(dist)[:remainder]
                data = slice_dict(data, crop_index)
                data["batch_index"] = len(batch) * np.ones_like(residue_index)
                batch.append(data)
                break
        result = {
            name: np.concatenate([d[name] for d in result], axis=0)
            for name in result[0]
        }
        result = pad_dict(result, self.size)
        return batch

class ProteinAtomPDB(AllPDB):
    def __init__(self, path, start_date="01/01/90", cutoff_date="12/31/21",
                 cutoff_resolution=4, filter_residue_type=None, seqres_aa="clusterSeqresAA",
                 seqres_na="clusterSeqresNA", assembly=True, split_types=False, npz_version=""):
        super().__init__(
            path, start_date, cutoff_date, cutoff_resolution,
            ["AA", "DNA", "RNA", "SMOL", "METAL"],
            seqres_aa, seqres_na, assembly,
            split_types, npz_version)
        self.aa_order = np.array(
            ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN',
             'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
             'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
             'TYR', 'VAL', 'UNK'])

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        pass # TODO

class ProteinSMOLNeighboursPDB(AllPDB):
    def __init__(self, path, start_date="01/01/90",
                 cutoff_date="12/31/21",
                 cutoff_resolution=4,
                 seqres_aa="clusterSeqresAA",
                 seqres_na="clusterSeqresNA",
                 assembly=True,
                 num_smol_neighbours=16,
                 p_atomize=0.02) -> None:
        super().__init__(
            path, start_date, cutoff_date,
            cutoff_resolution, ["AA", "DNA", "RNA", "SMOL", "METAL"],
            seqres_aa=seqres_aa,
            seqres_na=seqres_na,
            assembly=assembly, split_types=True,
            npz_version="_v3")
        self.p_atomize = p_atomize
        self.num_smol_neighbours = num_smol_neighbours
        self.aa_order = np.array(
            ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN',
             'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
             'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
             'TYR', 'VAL', 'UNK'])
        self.atom_type_order = np.array(['C', 'N', 'O', 'S', 'P'])

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        raw_data, chain = super().__getitem__("AA", index)
        # AA data
        aa_data = raw_data["AA"]
        atomize = np.random.rand(aa_data["residue_name"].shape[0]) < self.p_atomize
        raw_data["AASMOL"] = slice_dict(aa_data, atomize)
        aa_data = slice_dict(aa_data, ~atomize)
        aa_gt = np.argmax(
            aa_data["residue_name"][:, None] == self.aa_order[None, :], axis=-1)
        aa_gt = np.where(
            (aa_data["residue_name"][:, None] != self.aa_order[None, :]).all(axis=-1),
            20,
            aa_gt
        )
        residue_index = aa_data["residue_index"]
        chain_index = aa_data["chain_index"]
        entity_index = aa_data["entity_index"]
        all_atom_positions = aa_data["position"]
        all_atom_mask = aa_data["atom_mask"]

        # SMOL data
        smol_positions = []
        smol_types = []
        for smol in ["AASMOL", "DNA", "RNA", "SMOL", "METAL"]:
            na_data = raw_data[smol]
            atom_mask = na_data["atom_mask"]
            atom_positions = na_data["position"][atom_mask]
            atom_types = na_data["atom_type"][atom_mask]
            assignment = atom_types[:, None] == self.atom_type_order
            assignment = np.where(assignment.any(axis=-1), np.argmax(assignment, axis=-1), 6)
            if smol == "METAL":
                assignment = 5 * np.ones_like(assignment)
            smol_positions.append(atom_positions)
            smol_types.append(assignment)
        smol_positions = np.concatenate(smol_positions, axis=0)
        smol_types = np.concatenate(smol_types, axis=0)
        if False:
            ca = all_atom_positions[:, 1]
            distance = np.linalg.norm(ca[:, None] - smol_positions[None, :], axis=-1)
            neighbours = np.argsort(distance, axis=1)[:, :self.num_smol_neighbours]
            if neighbours.shape[1] < self.num_smol_neighbours:
                diff = self.num_smol_neighbours - neighbours.shape[1]
                neighbours = np.concatenate((
                    neighbours,
                    -np.ones((neighbours.shape[0], diff),
                            dtype=np.int32)), axis=1)
            smol_positions = smol_positions[neighbours]
            smol_types = smol_types[neighbours]
            smol_mask = neighbours != -1
            smol_positions = np.where(smol_mask[..., None], smol_positions, 0)
            smol_types = np.where(smol_mask, smol_types, 6)
        return dict(
            aa_gt=aa_gt,
            residue_index=residue_index,
            chain_index=chain_index,
            entity_index=entity_index,
            all_atom_positions=all_atom_positions,
            all_atom_mask=all_atom_mask,
            smol_positions=smol_positions,
            smol_types=smol_types,
            # smol_mask=smol_mask
        ), chain
    
    def __len__(self) -> int:
        return len(self.data_index["AA"])

def np_compute_pseudo_beta(x):
    n, ca, co = np.moveaxis(x[..., :3, :], -2, 0)
    b = ca - n
    c = co - ca
    a = np.cross(b, c)
    const = [-0.58273431, 0.56802827, -0.54067466]
    return const[0] * a + const[1] * b + const[2] * c + ca

class ProteinBinderPDB(ProteinPDB):
    def __init__(self, path, num_aa=1024, start_date="01/01/90", cutoff_date="12/31/21",
                 cutoff_resolution=4, seqres_aa="clusterSeqresAA", seqres_na="clusterSeqresNA",
                 assembly=True):
        super().__init__(
            path, start_date, cutoff_date, cutoff_resolution,
            seqres_aa, seqres_na, assembly)
        self.num_aa = num_aa

    def __getitem__(self, index) -> Dict[str, ndarray]:
        data, chain_id = super().__getitem__(index)
        chain = np.array([chain_id])[0]
        chain_index = data["chain_index"]
        chain_indices = np.unique(chain_index)
        # skip very large complexes as this absolutely
        # murders the cropping process
        if len(chain_indices) > 100:
            # print("too many chains")
            return None
        # discard if there is only one chain
        if len(chain_indices) < 2:
            # print("not a complex")
            return None
        selected_chain = chain_index == chain
        other_chains = chain_indices[chain_indices != chain]
        is_other = ~selected_chain
        binder_size = selected_chain.sum()
        # skip entities which are larger than the
        # desired maximum complex size
        if binder_size > self.num_aa:
            # print("binder too large")
            return None
        # skip empty entities
        if is_other.sum() == 0 or selected_chain.sum() == 0:
            # print("no protein found")
            return None
        is_target = ~selected_chain
        data["is_target"] = is_target
        # compute distances between in this chain vs others
        pos24 = data["all_atom_positions"]
        mask = data["all_atom_mask"]
        cb = np_compute_pseudo_beta(pos24)
        dist = np.linalg.norm(cb[is_other, None] - cb[None, selected_chain], axis=-1)
        dist = np.where(mask[is_other, None, 1] * mask[None, selected_chain, 1],
                        dist, np.inf)
        # print(dist.shape, index)
        dist = dist.min(axis=1)
        # get list of all chains with contacts to the selected chain
        has_any_contact = (dist < 6)
        chains_with_contact = np.unique(
            chain_index[is_other][has_any_contact])
        # set up hotspots for models to use
        hotspots = np.zeros_like(chain_index, dtype=np.bool_)
        hotspots[is_other] = has_any_contact
        data["hotspots"] = hotspots
        # if there are multiple chains but none with contacts
        # discard this entry
        if len(chains_with_contact) < 1:
            # print("no contacts")
            return None
        # select a number of random chains as the target chains
        chain_count = 1
        # guard against pathological randint from 1 to 2
        if len(chains_with_contact) > 1:
            chain_count = np.random.randint(1, len(chains_with_contact) + 1)
        np.random.shuffle(chains_with_contact)
        target_chains = chains_with_contact[:chain_count]
        selected_chains = np.array([chain] + list(target_chains))
        is_selected = (chain_index[:, None] == selected_chains[None, :]).any(axis=1)
        # slice everything by the selected chains
        data = slice_dict(data, is_selected)
        # crop to at most maximum size
        is_target = data["is_target"]
        
        remaining = self.num_aa - binder_size
        dist = dist[is_selected[is_other]]
        nearest_target = np.argsort(dist)[:min(remaining, 256)]
        nearest_mask = np.zeros((is_target.sum(),), dtype=np.bool_)
        nearest_mask[nearest_target] = True
        # keep the binder, of course
        keep = ~is_target
        # then, keep the nearest non-binder residues
        keep[is_target] = nearest_mask
        # slice everything to the crop
        data = slice_dict(data, keep)
        return data, chain_id

class ProteinCropPDB(ProteinPDB):
    def __init__(self, path, size, start_date="01/01/90", cutoff_date="12/31/21", cutoff_resolution=4,
                 seqres_aa="clusterSeqresAA", seqres_na="clusterSeqresNA", assembly=True) -> None:
        super().__init__(path, start_date, cutoff_date, cutoff_resolution, seqres_aa, seqres_na, assembly)
        self.size = size

    def __getitem__(self, index) -> Dict[str, ndarray]:
        raw_protein, chain = super().__getitem__(index)
        relevant_chain = raw_protein["chain_index"] == chain
        relevant_chain *= raw_protein["all_atom_mask"][:, 1]
        protein = slice_dict(raw_protein, relevant_chain)
        length = protein["aa_gt"].shape[0]
        if length > self.size:
            crop_start = np.random.randint(0, length - self.size)
            crop_end = crop_start + self.size
            protein = slice_dict(protein, slice(crop_start, crop_end))
        protein = pad_dict(protein, self.size)
        protein["batch_index"] = np.zeros((self.size,), dtype=np.int32)
        protein["chain_index"] = np.zeros((self.size,), dtype=np.int32)
        protein["mask"] = protein["all_atom_mask"].any(axis=-1)
        protein["seq_mask"] = protein["mask"] * (protein["aa_gt"] != 20)
        protein["residue_mask"] = protein["mask"] * protein["all_atom_mask"].any(axis=-1)
        return protein

class ProteinPDBSample(AllPDBSample):
    def __init__(self, path, start_date="01/01/90",
                 cutoff_date="12/31/21",
                 cutoff_resolution=4,
                 seqres_aa="clusterSeqresAA",
                 seqres_na="clusterSeqresNA",
                 assembly=True) -> None:
        super().__init__(
            path, start_date, cutoff_date,
            cutoff_resolution, ["AA"],
            seqres_aa=seqres_aa,
            seqres_na=seqres_na,
            assembly=assembly)
        self.format = "atom24"
        self.aa_order = np.array(
            ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN',
             'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
             'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
             'TYR', 'VAL', 'UNK'])
        self.atom_order_37 = np.array(
            ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
             'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
             'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
             'CZ3', 'NZ', 'OXT', ''])

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        data = super().__getitem__("AA", index)
        if data is None:
            return None
        raw_data, chain = data
        aa_gt = np.argmax(
            raw_data["residue_name"][:, None] == self.aa_order[None, :], axis=-1)
        aa_gt = np.where(
            (raw_data["residue_name"][:, None] != self.aa_order[None, :]).all(axis=-1),
            20,
            aa_gt
        )
        residue_index = raw_data["residue_index"]
        chain_index = raw_data["chain_index"]
        entity_index = raw_data["entity_index"]
        all_atom_positions = raw_data["position"]
        all_atom_mask = raw_data["atom_mask"]
        
        if self.format == "atom37":
            atom_name = raw_data["atom_name"]
            atom37_assignment = np.argmax(
                atom_name[:, :, None] == self.atom_order_37[None, None, :], axis=-2)
            idb = np.arange(atom_name.shape[0], dtype=np.int32)
            all_atom_positions = all_atom_positions[idb[:, None], atom37_assignment]
            all_atom_mask = all_atom_mask[idb[:, None], atom37_assignment]
        return dict(
            aa_gt=aa_gt,
            residue_index=residue_index,
            chain_index=chain_index,
            entity_index=entity_index,
            all_atom_positions=all_atom_positions,
            all_atom_mask=all_atom_mask
        ), chain
    
    def __len__(self) -> int:
        return len(self.data_index["AA"])

class BatchedProteinPDB(Dataset):
    def __init__(self, path, start_date="01/01/90",
                 cutoff_date="12/31/21",
                 cutoff_resolution=4,
                 size=1024, p_complex=0.5,
                 seqres_aa="clusterSeqresAA",
                 seqres_na="clusterSeqresNA",
                 assembly=True,
                 min_size=32,
                 max_size=None,
                 legacy_repetitive_chains=True,
                 base_dataset=ProteinPDB) -> None:
        super().__init__()
        self.protein_pdb = base_dataset(
            path, start_date, cutoff_date, cutoff_resolution,
            seqres_aa=seqres_aa,
            seqres_na=seqres_na,
            assembly=assembly)
        self.size = size
        self.min_size = min_size
        self.max_size = max_size or size
        self.p_complex = p_complex
        self.legacy_repetitive_chains = legacy_repetitive_chains
        self.current_index = []

    def get_next_pdb(self):
        data = None
        while data is None:
            if not self.current_index:
                print("shuffling...")
                self.current_index = list(range(len(self.protein_pdb)))
                random.shuffle(self.current_index)
            index = self.current_index.pop()
            data = self.protein_pdb[index]
        return data, index

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        total = 0
        result = []
        batch_index = []
        requeue = []
        count = 0
        num_iter = 0
        while total < self.size:
            num_iter += 1
            (protein_data, chain), dataset_index = self.get_next_pdb()
            # drop fully masked amino acids
            protein_data = slice_dict(protein_data, protein_data["all_atom_mask"].any(axis=-1))
            aa_gt = protein_data["aa_gt"]
            item_size = aa_gt.shape[0]

            # filter tiny entries
            if item_size < self.min_size:
                continue
            
            selected_chain = protein_data["chain_index"] == np.array([chain])[0]
            selected_chain_size = selected_chain.sum()

            # filter tiny chains
            if selected_chain_size < self.min_size or selected_chain_size > self.max_size:
                continue

            # filter chains which are too large
            if selected_chain_size > self.size:
                continue

            # filter repetitive chains
            # FIXME: this only works for monomers - for complexes
            # this results in discarding any complex where sufficient
            # amino acids are present in the other chains
            if self.legacy_repetitive_chains:
                # this is the original approach which has a bug for complexes
                _, aa_counts = np.unique(aa_gt, return_counts=True)
                repetitive = ((aa_counts / selected_chain_size) > 0.5).any()
            else:
                # this should be the correct approach to filter low-complexity
                # sequences
                _, aa_counts = np.unique(aa_gt[selected_chain], return_counts=True)
                repetitive = ((aa_counts / selected_chain_size) > 0.5).any()
            if repetitive:
                continue

            # skip according to length (as done in AlphaFold)
            skip_chance = 1 - max(min(selected_chain_size, 512), 256) / 512
            if random.random() < skip_chance:
                continue
            # skip chains wrongly marked as proteins
            if selected_chain_size == 0:
                continue
            # if the selected chain does not fit in the batch
            if total + selected_chain_size > self.size:
                requeue.append(dataset_index)
                # if the remaining size is large,
                # try the next protein
                if total < self.size / 2:
                    continue
                # otherwise, break
                else:
                    break
            # with probability 1-p_complex we're in single-sequence mode:
            # add just the chosen chain to the complex
            elif random.random() <= (1 - self.p_complex):
                # the selected chain fits in the batch
                part = slice_dict(protein_data, selected_chain)
            # otherwise, we're in complex mode:
            # add the entire complex if it fits
            elif total + item_size <= self.size:
                part = protein_data
            # if the single chain fits, but the complex as a whole
            # is too large:
            # add a subset of chains in the complex
            else:
                # add just the chosen chain to the batch
                part = slice_dict(protein_data, selected_chain)
            if part["chain_index"].shape[0] == 0:
                continue
            part["chain_index"] = numerical_chain_index(part["chain_index"])
            part_size = part["aa_gt"].shape[0]
            total += part_size
            batch_index += part_size * [count]
            count += 1
            result.append(part)
        # FIXME: requeue proteins skipped because of their size
        # self.current_index = requeue + self.current_index
        batch_index = np.array(batch_index, dtype=np.int32)
        result = {
            name: np.concatenate([d[name] for d in result], axis=0)
            for name in result[0]
        }
        result["batch_index"] = batch_index
        result = pad_dict(result, self.size)
        result["seq_mask"] = result["mask"] * (result["aa_gt"] != 20)
        result["residue_mask"] = result["mask"] * result["all_atom_mask"].any(axis=-1)

        return result
    
    def __len__(self):
        return len(self.protein_pdb)

# TODO refactor to reduce code duplication
class BatchedProteinPDBStream(IterableDataset):
    def __init__(self, path, start_date="01/01/90",
                 cutoff_date="12/31/21",
                 cutoff_resolution=4,
                 size=1024, p_complex=0.5,
                 seqres_aa="clusterSeqresAA",
                 seqres_na="clusterSeqresNA",
                 assembly=True,
                 min_size=32,
                 max_size=None,
                 legacy_repetitive_chains=True,
                 base_dataset=ProteinPDB) -> None:
        super().__init__()
        self.protein_pdb = base_dataset(
            path, start_date, cutoff_date, cutoff_resolution,
            seqres_aa=seqres_aa,
            seqres_na=seqres_na,
            assembly=assembly)
        self.size = size
        self.min_size = min_size
        self.max_size = max_size or size
        self.p_complex = p_complex
        self.legacy_repetitive_chains = legacy_repetitive_chains
        self.current_index = []

    def get_next_pdb(self):
        data = None
        while data is None:
            if not self.current_index:
                self.current_index = list(range(len(self.protein_pdb)))
                random.shuffle(self.current_index)
            index = self.current_index.pop()
            data = self.protein_pdb[index]
        return data, index

    def next_item(self) -> Dict[str, np.ndarray]:
        total = 0
        result = []
        batch_index = []
        requeue = []
        count = 0
        num_iter = 0
        while total < self.size:
            num_iter += 1
            (protein_data, chain), dataset_index = self.get_next_pdb()
            # drop fully masked amino acids
            # FIXME: data needs to have all backbone atoms, otherwise it's dropped
            has_smol = False
            if "smol_types" in protein_data:
                has_smol = True
                smol_data = dict(
                    smol_types=protein_data["smol_types"],
                    smol_positions=protein_data["smol_positions"])
                protein_data = {key: value for key, value in protein_data.items() if not key.startswith("smol")}
            protein_data = slice_dict(protein_data, protein_data["all_atom_mask"][:, :3].all(axis=-1))
            aa_gt = protein_data["aa_gt"]
            item_size = aa_gt.shape[0]

            # filter tiny entries
            if item_size < self.min_size:
                continue
            
            selected_chain = protein_data["chain_index"] == np.array([chain])[0]
            selected_chain_size = selected_chain.sum()

            # filter tiny chains
            if selected_chain_size < self.min_size or selected_chain_size > self.max_size:
                continue

            # filter chains which are too large
            if selected_chain_size > self.size:
                continue

            # filter repetitive chains
            if self.legacy_repetitive_chains:
                _, aa_counts = np.unique(aa_gt, return_counts=True)
            else:
                _, aa_counts = np.unique(aa_gt[selected_chain], return_counts=True)
            repetitive = ((aa_counts / selected_chain_size) > 0.5).any()
            if repetitive:
                continue

            # skip according to length (as done in AlphaFold)
            skip_chance = 1 - max(min(selected_chain_size, 512), 256) / 512
            if random.random() < skip_chance:
                continue
            # skip chains wrongly marked as proteins
            if selected_chain_size == 0:
                continue
            # if the selected chain does not fit in the batch
            if total + selected_chain_size > self.size:
                requeue.append(dataset_index)
                # if the remaining size is large,
                # try the next protein
                if total < self.size / 2:
                    continue
                # otherwise, break
                else:
                    break
            # with probability 1-p_complex we're in single-sequence mode:
            # add just the chosen chain to the complex
            elif random.random() <= (1 - self.p_complex):
                # the selected chain fits in the batch
                part = slice_dict(protein_data, selected_chain)
            # otherwise, we're in complex mode:
            # add the entire complex if it fits
            elif total + item_size <= self.size:
                part = protein_data
            # if the single chain fits, but the complex as a whole
            # is too large:
            # add a subset of chains in the complex
            else:
                # add just the chosen chain to the batch
                part = slice_dict(protein_data, selected_chain)
            if part["chain_index"].shape[0] == 0:
                continue
            part["chain_index"] = numerical_chain_index(part["chain_index"])
            part_size = part["aa_gt"].shape[0]
            if has_smol:
                smol_positions = smol_data["smol_positions"]
                smol_types = smol_data["smol_types"]
                ca = part["all_atom_positions"][:, 1]
                distance = np.linalg.norm(ca[:, None] - smol_positions[None, :], axis=-1)
                neighbours = np.argsort(distance, axis=1)[:, :16]
                if neighbours.shape[1] < 16:
                    diff = 16 - neighbours.shape[1]
                    neighbours = np.concatenate((
                        neighbours,
                        -np.ones((neighbours.shape[0], diff),
                                dtype=np.int32)), axis=1)
                if smol_positions.shape[0] == 0:
                    smol_positions = np.zeros(list(neighbours.shape) + [3], dtype=np.float32)
                    smol_types = np.zeros(neighbours.shape, dtype=np.int32)
                    smol_mask = np.zeros(neighbours.shape, dtype=np.bool_)
                else:
                    smol_positions = smol_positions[neighbours]
                    smol_types = smol_types[neighbours]
                    smol_mask = neighbours != -1
                    smol_positions = np.where(smol_mask[..., None], smol_positions, 0)
                    smol_types = np.where(smol_mask, smol_types, 6)
                part["smol_positions"] = smol_positions
                part["smol_types"] = smol_types
                part["smol_mask"] = smol_mask
            total += part_size
            batch_index += part_size * [count]
            count += 1
            result.append(part)
        self.current_index = self.current_index + requeue # FIXME: pop pops from end?
        batch_index = np.array(batch_index, dtype=np.int32)
        result = {
            name: np.concatenate([d[name] for d in result], axis=0)
            for name in result[0]
        }
        result["batch_index"] = batch_index
        result = pad_dict(result, self.size)
        result["seq_mask"] = result["mask"] * (result["aa_gt"] != 20)
        result["residue_mask"] = result["mask"] * result["all_atom_mask"].any(axis=-1)

        return result

    def __iter__(self):
        while True:
            yield self.next_item()

class CroppedPDBStream(BatchedProteinPDBStream):
    def __init__(self, path, start_date="01/01/90", cutoff_date="12/31/21",
                 cutoff_resolution=4, size=1024, p_complex=0.5,
                 seqres_aa="clusterSeqresAA", seqres_na="clusterSeqresNA",
                 assembly=True, min_size=32, max_size=None, spatial_crop_monomer=0.0,
                 legacy_repetitive_chains=True,
                 base_dataset=ProteinPDB):
        super().__init__(path, start_date, cutoff_date, cutoff_resolution,
                         size, p_complex, seqres_aa, seqres_na, assembly,
                         min_size, max_size, legacy_repetitive_chains,
                         base_dataset)
        self.spatial_crop_monomer = spatial_crop_monomer

    def next_item(self) -> Dict[str, np.ndarray]:
        while True:
            (protein_data, chain), _ = self.get_next_pdb()
            protein_data = slice_dict(protein_data, protein_data["all_atom_mask"][:, :3].all(axis=-1))
            aa_gt = protein_data["aa_gt"]
            item_size = aa_gt.shape[0]

            # filter tiny entries
            if item_size < self.min_size:
                continue
            
            selected_chain = protein_data["chain_index"] == np.array([chain])[0]
            selected_chain_size = selected_chain.sum()

            # filter tiny chains
            if selected_chain_size < self.min_size:
                continue

            # skip according to length (as done in AlphaFold)
            skip_chance = 1 - max(min(selected_chain_size, 512), 256) / 512
            if random.random() < skip_chance:
                continue
            # skip chains wrongly marked as proteins
            if selected_chain_size == 0:
                continue

            # crop chain to maximum size
            protein_data = self.crop(protein_data, chain)
            protein_data["chain_index"] = numerical_chain_index(protein_data["chain_index"])
            protein_data["batch_index"] = np.zeros_like(protein_data["chain_index"])
            protein_data["seq_mask"] = protein_data["mask"] * (protein_data["aa_gt"] != 20)
            protein_data["residue_mask"] = protein_data["seq_mask"] * protein_data["all_atom_mask"].any(axis=-1)
            return protein_data

    def crop(self, data, chain):
        chain_index = data["chain_index"]
        selected_chain = chain_index == np.array([chain])[0]
        selected_chain_size = selected_chain.sum()
        chains = np.unique(chain_index)
        np.random.shuffle(chains)
        num_chains = chains.shape[0]
        # return a complex crop with probability p_complex
        p_spatial = 0.5
        if num_chains > 1 and random.random() < self.p_complex:
            # print("complex crop...")
            # interface spatial crop
            if random.random() < p_spatial:
                # print("spatial crop...")
                ca = data["all_atom_positions"][:, 1]
                sel_ca = ca[selected_chain]
                other = ca[~selected_chain]
                contact = (np.linalg.norm(sel_ca[:, None] - other[None, :], axis=-1) < 8).any(axis=1)
                center = np.argmax(contact * np.random.rand(*contact.shape))
                sel_ca = ca[selected_chain][center]
                selected = np.argsort(np.linalg.norm(ca - sel_ca, axis=-1), axis=0)[:self.size]
                mask = np.zeros_like(selected_chain)
                mask[selected] = True
                data = slice_dict(data, mask)
            # contiguous chain crop
            else:
                # print("segment crop...")
                budget = self.size
                mask = np.zeros_like(selected_chain, dtype=np.bool_)
                for idx, chain in enumerate(chains):
                    # print("budged", budget)
                    # define the mask of residues for the chain
                    chain_slice = chain_index == chain
                    # get the length of the chain
                    chain_length = chain_slice.astype(np.int32).sum()
                    # compute the minimum and maximum lengths
                    # of a chain-crop
                    min_length = max(min(min(16, chain_length), budget), 0)
                    max_length = max(min(chain_length, budget), 0)
                    if (max_length < 16) or (min_length == max_length):
                        # print("failure lengths", chain_length, min_length, max_length)
                        break
                    # sample a random length in this range and define
                    # a random crop
                    length = random.randint(min_length, max_length)
                    # if this is the last iteration, make sure to fill
                    # the batch
                    if idx == len(chains) - 1:
                        length = min(chain_length, budget - chain_length)
                    start = random.randint(0, chain_length - length)
                    end = start + length
                    # add the defined crop to the selection mask
                    mask[np.nonzero(chain_slice)[0][start:end]] = True
                    # and reduce the remaining budget by its length
                    budget -= length
                data = slice_dict(data, mask)                
        # otherwise, return a contiguous crop of a single chain
        else:
            # print("monomer crop...")
            data = slice_dict(data, selected_chain)
            if selected_chain_size > self.size:
                if random.random() < self.spatial_crop_monomer:
                    residue = random.randrange(0, selected_chain_size)
                    ca = data["pos"][:, 1]
                    residue_ca = ca[residue]
                    nearest = np.argsort(np.linalg.norm(residue_ca - ca, axis=-1), axis=0)[:self.size]
                    data = slice_dict(data, nearest)
                else:
                    start = random.randrange(0, selected_chain_size - self.size)
                    end = start + self.size
                    data = slice_dict(data, slice(start, end))
            # print("start size", selected_chain_size)
        data = pad_dict(data, self.size)
        # print("crop size", data["mask"].sum())
        return data

class BinderPDBStream(IterableDataset):
    def __init__(self, path, start_date="01/01/90",
                 cutoff_date="12/31/21",
                 cutoff_resolution=4,
                 size=1024, p_complex=0.5,
                 seqres_aa="clusterSeqresAA",
                 seqres_na="clusterSeqresNA",
                 assembly=True,
                 min_size=32,
                 max_size=None) -> None:
        super().__init__()
        self.protein_pdb = ProteinBinderPDB(
            path, size,
            start_date, cutoff_date, cutoff_resolution,
            seqres_aa=seqres_aa,
            seqres_na=seqres_na,
            assembly=assembly)
        self.size = size
        self.min_size = min_size
        self.max_size = max_size or size
        self.p_complex = p_complex
        self.current_index = []

    def get_next_pdb(self):
        data = None
        while data is None:
            # print("attempting next PDB...")
            if not self.current_index:
                self.current_index = list(range(len(self.protein_pdb)))
                random.shuffle(self.current_index)
            index = self.current_index.pop()
            data = self.protein_pdb[index]
        return data, index

    def next_item(self) -> Dict[str, np.ndarray]:
        total = 0
        result = []
        batch_index = []
        requeue = []
        count = 0
        num_iter = 0
        while total < self.size:
            # print("loading next item...")
            num_iter += 1
            (protein_data, chain), dataset_index = self.get_next_pdb()
            protein_data = slice_dict(protein_data, protein_data["all_atom_mask"][:, :3].all(axis=-1))
            aa_gt = protein_data["aa_gt"]
            item_size = aa_gt.shape[0]

            # filter tiny entries
            if item_size < self.min_size:
                # print("too small")
                continue
            
            selected_chain = protein_data["chain_index"] == np.array([chain])[0]
            selected_chain_size = selected_chain.sum()

            # filter tiny chains
            if selected_chain_size < self.min_size or selected_chain_size > self.max_size:
                # print("chain out of size")
                continue

            # filter chains which are too large
            if selected_chain_size > self.size:
                # print("chain too large")
                continue

            # filter repetitive chains
            _, aa_counts = np.unique(aa_gt[selected_chain], return_counts=True)
            repetitive = ((aa_counts / selected_chain_size) > 0.5).any()
            if repetitive:
                repmax = (aa_counts / selected_chain_size).max()
                # print(f"chain repetitive at maximum {repmax:.2f}")
                # print(decode_sequence(aa_gt))
                continue

            # skip according to length (as done in AlphaFold)
            # FIXME: skip chance considered harmful
            # skip_chance = 1 - max(min(selected_chain_size, 512), 256) / 512
            # if random.random() < skip_chance:
            #     print("chain skipped")
            #     continue
            # skip chains wrongly marked as proteins
            if selected_chain_size == 0:
                # print("chain empty")
                continue
            # if the selected chain does not fit in the batch
            if total + item_size > self.size:
                requeue.append(dataset_index)
                # if the remaining size is large,
                # try the next protein
                if total < self.size / 2:
                    # print("doesn't fit, requeue and keep going")
                    continue
                # otherwise, break
                else:
                    # print("doesn't fit, requeue and break")
                    break
            # with probability 1-p_complex we're in single-sequence mode:
            # add just the chosen chain to the complex
            # elif random.random() <= (1 - self.p_complex):
            #     # the selected chain fits in the batch
            #     part = slice_dict(protein_data, selected_chain)
            # otherwise, we're in complex mode:
            # add the entire complex if it fits
            else:
                part = protein_data
            # if the single chain fits, but the complex as a whole
            # is too large:
            # add a subset of chains in the complex
            # else:
            #     # add just the chosen chain to the batch
            #     part = slice_dict(protein_data, selected_chain)
            if part["chain_index"].shape[0] == 0:
                # print("part empty")
                continue
            part["chain_index"] = numerical_chain_index(part["chain_index"])
            part_size = part["aa_gt"].shape[0]
            total += part_size
            batch_index += part_size * [count]
            count += 1
            result.append(part)
        self.current_index = self.current_index + requeue
        batch_index = np.array(batch_index, dtype=np.int32)
        result = {
            name: np.concatenate([d[name] for d in result], axis=0)
            for name in result[0]
        }
        result["batch_index"] = batch_index
        result = pad_dict(result, self.size)
        result["seq_mask"] = result["mask"] * (result["aa_gt"] != 20)
        result["residue_mask"] = result["mask"] * result["all_atom_mask"].any(axis=-1)

        return result

    def __iter__(self):
        while True:
            yield self.next_item()

def numerical_chain_index(chain):
    indices = np.unique(chain)
    return np.argmax(chain[:, None] == indices[None, :], axis=-1)

def slice_dict(data, indices):
    return {
        name: item[indices] if item.ndim > 0 else item
        for name, item in data.items()
        if name not in ["has_structure"]
    }

def pad_item(data, target_size):
  r"""Pad a data tensor to the desired crop size, if
  it is shorter.
  
  Args:
    data (np.ndarray): Data to be padded.
  
  Returns:
    result (np.ndarray): zero-padded result tensor.
    mask (np.ndarray): mask filled with 1 for non-padding data.
  """
  if not isinstance(data, np.ndarray):
    return data, None
  size = data.shape[0]
  result = data
  mask = np.ones(target_size, np.bool_)
  mask[size:] = 0
  if size < target_size:
    diff = target_size - size
    padding = np.zeros([diff] + list(data.shape[1:]), dtype=data.dtype)
    result = np.concatenate((data, padding), axis=0)
  return result, mask

def pad_dict(data, target_size):
    result = {}
    mask = None
    for name in data:
        result[name], mask = pad_item(data[name], target_size)
    result["mask"] = mask
    return result

def parse_date(date):
    month, day, year = date.split("/")
    day = int(day)
    month = int(month)
    year = int(year)
    if year >= 30:
        year += 1900
    else:
        year += 2000
    return datetime.date(year=year, month=month, day=day)

def parse_resolution(resolution):
    if resolution in ("NOT", ""):
        return 999.0
    if "," in resolution:
        return max(*map(parse_resolution, resolution.split(",")))
    return float(resolution)

def read_entries(entries_path):
    entry_dict = dict()
    with open(entries_path, "rt") as f:
        fit = iter(f)
        header = next(fit)
        separator = next(fit)
        for line in fit:
            code, _, date, *_, resolution, exptype = line.strip().split("\t")
            date = parse_date(date)
            resolution = parse_resolution(resolution)
            entry_dict[code] = dict(
                date=date,
                resolution=resolution,
                exptype=exptype)
    return entry_dict

def read_clusters(cluster_path, clusters=None, entry_clusters=None):
    clusters = clusters or {}
    entry_clusters = entry_clusters or {}
    with open(cluster_path, "rt") as f:
        for line in f:
            representative, entry = line.strip().split("\t")
            representative_base = representative.split("_")[0].upper()
            entry_base = entry.split("_")[0].upper()
            if representative not in clusters:
                clusters[representative] = set()
            if representative_base not in entry_clusters:
                entry_clusters[representative_base] = dict()
            clusters[representative].add(entry)
            if entry_base not in entry_clusters[representative_base]:
                entry_clusters[representative_base][entry_base] = dict()
    for cluster, members in clusters.items():
        cluster_base = cluster.split("_")[0].upper()
        for member in members:
            member_base = member.split("_")[0].upper()
            if cluster not in entry_clusters[cluster_base][member_base]:
                entry_clusters[cluster_base][member_base][cluster] = []
            entry_clusters[cluster_base][member_base][cluster].append(member)
    return clusters, entry_clusters

def create_entry_index(entry_clusters, entries, npz_assemblies, filter):
    result = []
    for cluster, members in entry_clusters.items():
        filtered_cluster = []
        for member in members:
            metadata = entries[member]
            if filter(metadata) and (member in npz_assemblies):
                filtered_cluster.append(dict(member=member,
                                             assemblies=[
                                                assembly
                                                for assembly in npz_assemblies[member]
                                             ],
                                             clusters=[
                                                dict(
                                                    subcluster=subcluster,
                                                    members=submembers)
                                                for subcluster, submembers in members[member].items()],
                                             metadata=metadata))
        result.append(filtered_cluster)
    return result

def read_chain_assemblies(path):
    with open(path, "rt") as f:
        result = {}
        for line in f:
            pdbid, chain, assembly = line.strip().split(",")
            key = pdbid, chain
            if key not in result:
                result[key] = []
            result[key].append(assembly)
    return result

def create_chain_index(chain_clusters, entries, npz_assemblies, chain_assemblies, filter):
    result = []
    for _, members in chain_clusters.items():
        filtered_cluster = []
        for member in members:
            member_base = member.split("_")[0].upper()
            chain = member.split("_")[1]
            if member_base not in entries:
                continue # skip obsolete structures
            metadata = entries[member_base]
            if chain_assemblies is not None:
                has_chain = ((member_base, chain) in chain_assemblies)
            else:
                has_chain = True
            if filter(metadata) and has_chain:
                if chain_assemblies is not None:
                    this_assemblies = chain_assemblies[member_base, chain]
                else:
                    this_assemblies = npz_assemblies[member_base]
                if not this_assemblies:
                    # skip chains without assemblies
                    # these can occur if all assemblies for
                    # a given chain have corrupted input files
                    continue
                filtered_cluster.append(
                    dict(entry=member_base,
                         chain=chain,
                         assemblies=this_assemblies,
                         metadata=metadata))
        if filtered_cluster:
            result.append(filtered_cluster)
    return result

def create_cluster_weights(index):
    result = []
    for filtered_cluster in index:
        cluster_size = len(filtered_cluster)
        weight = 1 / cluster_size
        for member in filtered_cluster:
            member["weight"] = weight
            result.append(member)
    return result

def compute_index(path, start_date="12/31/90", cutoff_date="12/31/21", cutoff_resolution=4.0,
                  seqres_aa="clusterSeqresAA", seqres_na="clusterSeqresNA",
                  npz_version=""):
    def entry_filter(x):
        return (
            x["resolution"] <= cutoff_resolution
            and x["date"] >= parse_date(start_date)
            and x["date"] < parse_date(cutoff_date))

    npz_path = f"{path}/assembly{npz_version}_npz/"
    entry_path = f"{path}/entries.idx"
    cluster_aa = f"{path}/{seqres_aa}_cluster.tsv"
    cluster_na = f"{path}/{seqres_na}_cluster.tsv"
    chain_assemblies = read_chain_assemblies(f"{path}/chain_assemblies.csv")
    subfolders_npz = os.listdir(npz_path)
    available_npz = [
        name
        for subfolder in subfolders_npz
        for name in os.listdir(f"{npz_path}/{subfolder}")
    ]
    npz_assemblies = {}
    for name in available_npz:
        basename = name.split("-")[0].upper()
        if not basename in npz_assemblies:
            npz_assemblies[basename] = []
        npz_assemblies[basename].append(f"{npz_path}/{name[1:3]}/{name}")
    entries = read_entries(entry_path)
    aa_clusters, _ = read_clusters(cluster_aa)
    na_clusters, _ = read_clusters(cluster_na)
    aa_index = create_chain_index(aa_clusters, entries, npz_assemblies, chain_assemblies, entry_filter)
    na_index = create_chain_index(na_clusters, entries, npz_assemblies, chain_assemblies, entry_filter)
    return dict(AA=aa_index, NA=na_index)

def compute_asym_index(path, start_date="12/31/90", cutoff_date="12/31/21", cutoff_resolution=4.0,
                       seqres_aa="clusterSeqresAA", seqres_na="clusterSeqresNA"):
    def entry_filter(x):
        return (
            x["resolution"] <= cutoff_resolution
            and x["date"] >= parse_date(start_date)
            and x["date"] < parse_date(cutoff_date))

    npz_path = f"{path}/asym_npz/"
    entry_path = f"{path}/entries.idx"
    cluster_aa = f"{path}/{seqres_aa}_cluster.tsv"
    cluster_na = f"{path}/{seqres_na}_cluster.tsv"
    subfolders_npz = os.listdir(npz_path)
    available_npz = [
        name
        for subfolder in subfolders_npz
        for name in os.listdir(f"{npz_path}/{subfolder}")
    ]
    npz_assemblies = {}
    for name in available_npz:
        basename = name[:4].upper()
        if not basename in npz_assemblies:
            npz_assemblies[basename] = []
        npz_assemblies[basename].append(f"asym_npz/{name[1:3]}/{name}")
    entries = read_entries(entry_path)
    aa_clusters, _ = read_clusters(cluster_aa)
    na_clusters, _ = read_clusters(cluster_na)
    aa_index = create_chain_index(aa_clusters, entries, npz_assemblies, None, entry_filter)
    na_index = create_chain_index(na_clusters, entries, npz_assemblies, None, entry_filter)
    return dict(AA=aa_index, NA=na_index)
