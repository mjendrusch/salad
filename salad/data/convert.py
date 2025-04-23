import numpy as np
import gemmi

# open chemical component dictionary
DNA_BASES = ["DG", "DA", "DT", "DC"]
RNA_BASES = ["G", "A", "U", "C", "I"]
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS",
               "GLN", "GLU", "GLY", "HIS", "ILE",
               "LEU", "LYS", "MET", "PHE", "PRO",
               "SER", "THR", "TRP", "TYR", "VAL",
               "PYL", "SEC", "UNK"]
METALS = ["MG", "ZN", "FE"]
BOND_TYPES = ["None", "Single", "Double", "Triple", "Aromatic"]
AA_DICT = {}

def read_components(path):
    return gemmi.cif.read(path)

def get_residue_info(chemcomp, name, max_bonds=12):
    block = chemcomp.find_block(name)
    if block is None:
        raise ValueError("Invalid residue...")
    residue = gemmi.make_chemcomp_from_block(block)
    restruct = None
    for model in gemmi.make_structure_from_chemcomp_block(block):
        restruct = model.find_chain("")[0]
        break
    pos = None
    if restruct is not None:
        pos = np.array(
            [[atom.pos.x, atom.pos.y, atom.pos.z]
             for atom in restruct
             if not atom.is_hydrogen()
            ])
    atom_order = []
    atom_types = []
    atom_is_hydrogen = set()
    for atom in residue.atoms:
        if atom.el.name != "H":
            atom_order.append(atom.id)
            atom_types.append(atom.el.name)
        else:
            atom_is_hydrogen.add(atom.id)
    atom_order = atom_order
    atom_types = atom_types
    bond_index = np.zeros((len(atom_order), max_bonds), dtype=np.int32) - 1
    bond_type = np.zeros((len(atom_order), max_bonds), dtype=np.int32)
    has_bond = np.zeros((len(atom_order), max_bonds), dtype=np.bool_)
    hydrogens = np.zeros((len(atom_order)), dtype=np.int32)
    for bond in residue.rt.bonds:
        next_bond = has_bond.astype(np.int32).sum(axis=1)
        if bond.id1.atom in atom_is_hydrogen and bond.id2.atom in atom_order:
            hydrogens[atom_order.index(bond.id2.atom)] += 1
        elif bond.id2.atom in atom_is_hydrogen and bond.id1.atom in atom_order:
            hydrogens[atom_order.index(bond.id1.atom)] += 1
        elif bond.id1.atom in atom_order and bond.id2.atom in atom_order:
            id1 = atom_order.index(bond.id1.atom)
            id2 = atom_order.index(bond.id2.atom)
            kind = BOND_TYPES.index(bond.type.name)
            kind = BOND_TYPES.index("Aromatic") if bond.aromatic else kind
            if not ((id1 == bond_index[id2]).any() or (id2 == bond_index[id1]).any()):
                bond_index[id1, next_bond[id1]] = id2
                bond_index[id2, next_bond[id2]] = id1
                bond_type[id1, next_bond[id1]] = kind
                bond_type[id2, next_bond[id2]] = kind
                has_bond[id1, next_bond[id1]] = True
                has_bond[id2, next_bond[id2]] = True
    return dict(
        atom_order=atom_order,
        atom_order_index=np.arange(len(atom_order)),
        atom_types=atom_types,
        bond_index=bond_index,
        bond_type=bond_type,
        has_bond=has_bond,
        hydrogens=hydrogens,
        positions=pos
    )


def parse_structure(components, path, max_chains=50, get_bfactor=False):
    structure = gemmi.read_structure(path)
    uresi = 0
    ustsi = 0
    bond_index = []
    bond_type = []
    has_bond = []
    res_lengths = []
    res_names = []
    entity_index = []
    chain_index = []
    molecule_index = []
    residue_index = []
    unique_standard_index = []
    residue_type = []
    atom_names = []
    atom_types = []
    atom_bfactor = []
    atom_order_index = []
    positions = []
    atom_mask = []
    mol_id_list = []
    residues = dict()
    hoh_count = 0
    for model in structure:
        for idx, chain in enumerate(model):
            if len(model) > max_chains and "-" in chain.name:
                continue
            for subchain in chain.subchains():
                for res in subchain:
                    try:
                        res_info = get_residue_info(components, res.name)
                    except ValueError as e:
                        print(f"INVALID RESIDUE {res.name}")
                        continue
                    if res.name not in residues:
                        residues[res.name] = res_info
                    resi = res.label_seq
                    if resi is None:
                        resi = res.seqid.num
                        if resi is None:
                            resi = -1
                    if res.name == "HOH":
                        mol_id = f"HOH-{hoh_count}"
                        hoh_count += 1
                        restype = "HOH"
                    elif res.name in AMINO_ACIDS:
                        restype = "AA"
                    elif res.name in DNA_BASES:
                        restype = "DNA"
                    elif res.name in RNA_BASES:
                        restype = "RNA"
                    elif res.name in METALS:
                        restype = "METAL"
                    else:
                        restype = "SMOL"
                    raw_mol_id = subchain.subchain_id()
                    if raw_mol_id in mol_id_list:
                        mol_id = mol_id_list.index(raw_mol_id)
                    else:
                        mol_id = len(mol_id_list)
                        mol_id_list.append(raw_mol_id)
                    atom_order = res_info["atom_order"]
                    local_atom_mask = []
                    res_atoms = {}
                    true_length = 0
                    for atom in res:
                        atom_name = atom.name
                        if atom_name not in res_atoms:
                            res_atoms[atom_name] = {}
                            res_atoms[atom_name]["element"] = atom.element.name
                            res_atoms[atom_name]["position"] = np.array((atom.pos.x, atom.pos.y, atom.pos.z))
                            if get_bfactor:
                                res_atoms[atom_name]["bfactor"] = atom.b_iso
                    for aoi, atom_name in enumerate(atom_order):
                        if atom_name in res_atoms:
                            atom_names.append(atom_name)
                            atom_order_index.append(aoi)
                            if get_bfactor:
                                atom_bfactor.append(res_atoms[atom_name]["bfactor"])
                            atom_types.append(res_atoms[atom_name]["element"])
                            positions.append(res_atoms[atom_name]["position"])
                            bond_index.append(res_info["bond_index"][aoi])
                            bond_type.append(res_info["bond_type"][aoi])
                            has_bond.append(res_info["has_bond"][aoi])
                            atom_mask.append(True)
                            local_atom_mask.append(True)
                            true_length += 1
                        elif restype not in ("METAL", "SMOL"):
                            atom_names.append(atom_name)
                            atom_order_index.append(aoi)
                            if get_bfactor:
                                atom_bfactor.append(0.0)
                            atom_types.append("")
                            positions.append(np.array((0.0, 0.0, 0.0)))
                            bond_index.append(res_info["bond_index"][aoi])
                            bond_type.append(res_info["bond_type"][aoi])
                            has_bond.append(res_info["has_bond"][aoi])
                            atom_mask.append(False)
                            local_atom_mask.append(True)
                            true_length += 1
                        else:
                            local_atom_mask.append(False)
                            continue
                        residue_type.append(restype)
                        residue_index.append(resi)
                        unique_standard_index.append(ustsi)
                        if restype == "SMOL":
                            ustsi += 1
                        res_names.append(res.name)
                        res_entity_id = res.entity_id
                        if not res_entity_id:
                            res_entity_id = 0
                        entity_index.append(res_entity_id)
                        chain_index.append(chain.name)
                        molecule_index.append(mol_id)
                    res_lengths += true_length * [true_length]
                    local_atom_mask = np.array(local_atom_mask, dtype=np.bool_)
                    uresi += 1
                    if restype != "SMOL":
                        ustsi += 1
    atom_names = np.array(atom_names)
    atom_types = np.array(atom_types)
    atom_mask = np.array(atom_mask, dtype=np.bool_)
    atom_order_index = np.array(atom_order_index)
    atom_bfactor = np.array(atom_bfactor, dtype=np.float32)
    positions = np.array(positions)
    entity_index = np.array(entity_index, dtype=np.int32)
    chain_index = np.array(chain_index)
    residue_index = np.array(residue_index)
    residue_name = np.array(res_names)
    unique_standard_index = np.array(unique_standard_index)
    residue_type = np.array(residue_type)
    molecule_index = np.array(molecule_index)
    is_hoh = residue_type == "HOH"
    max_molecule_index = molecule_index.max()
    molecule_index = np.where(
        is_hoh,
        max_molecule_index + np.arange(is_hoh.shape[0]),
        molecule_index)
    residue_atom_count = np.array(res_lengths)
    bond_index = np.stack(bond_index, axis=0)
    bond_type = np.stack(bond_type, axis=0)
    has_bond = np.stack(has_bond, axis=0)
    parse_result = dict(
        atom_name=atom_names,
        atom_type=atom_types,
        atom_mask=atom_mask,
        atom_order_index=atom_order_index,
        position=positions,
        entity_index=entity_index,
        chain_index=chain_index,
        residue_index=residue_index,
        residue_name=residue_name,
        unique_standard_index=unique_standard_index,
        residue_type=residue_type,
        molecule_index=molecule_index,
        residue_atom_count=residue_atom_count,
        bond_index=bond_index,
        bond_type=bond_type,
        has_bond=has_bond
    )
    if get_bfactor:
        parse_result["bfactor"] = atom_bfactor
    return to_atomX(parse_result, 24)

def to_atomX(target, max_count=50):
    residue_indices, length = np.unique(target["unique_standard_index"], return_counts=True)
    residue_count = residue_indices.shape[0]
    in_mask = np.arange(max_count)[None, :] < length[:, None] 
    resdict = dict()
    for name, item in target.items():
        resdict[name] = np.zeros([residue_count, max_count] + list(item.shape[1:]), dtype=item.dtype)
        resdict[name][in_mask] = item
    for name in ("residue_type", "residue_name", "residue_index",
                 "chain_index", "unique_standard_index", "molecule_index",
                 "entity_index", "residue_atom_count"):
        resdict[name] = resdict[name][:, 0]
    return resdict
