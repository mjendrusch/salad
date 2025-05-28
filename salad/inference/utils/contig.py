"""Parsers for RFdiffusion-style contig strings for multi-motif specification.

Scaffolding motifs from two PDB files motif_1.pdb and motif_2.pdb can be done
using a specification like this:

"motif_1.pdb, motif_2.pdb -> 20 / pdb = 0, chain = A, group = 0, 1-20 / 50 / pdb = 1, chain = A, group = 1, 50-75 / 20"

This will produce a structure with the following blocks:
[20 random residues]
[residues 1-20 in chain A of motif_1.pdb]
[50 random residues]
[residues 50-75 in chain A of motif_2.pdb]
[20 random residues]
"""


from typing import List

import numpy as np

from salad.aflib.model.geometry import Vec3Array
from salad.aflib.common.protein import from_pdb_string, Protein
from salad.aflib.model.all_atom_multimer import atom37_to_atom14

from salad.data.allpdb import np_to_ncacocb

def parse_contig(data: str):
    """Parses a RFdiffusion-style contig specification.
    
    Contig specifications have the following syntax:
    CONTIG      := COMPLEX | PATH_LIST "->" COMPLEX
    COMPLEX     := CHAIN | CHAIN ":" COMPLEX
    CHAIN       := ASSIGN_LIST | ASSIGN_LIST "/" CHAIN
    ASSIGN      := NAME "=" (INT | STR)
    ASSIGN_LIST := INT | RANGE | ASSIGN | ASSIGN "," ASSIGN_LIST
    RANGE       := INT "-" INT
    """
    pdb_paths, assembly = parse_contig_expr(data)
    pdb_files = [_parse_pdb(path) for path in pdb_paths]
    data = interpret_contig_expr(assembly, pdb_files)
    return data

def _parse_pdb(path):
    with open(path) as f:
        protein: Protein = from_pdb_string(f.read(), convert_chains=False)
        atom_positions, atom_mask = atom37_to_atom14(
            protein.aatype,
            Vec3Array.from_array(protein.atom_positions),
            protein.atom_mask)
        atom_positions = np.array(atom_positions.to_array())
        return Protein(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=protein.aatype,
            residue_index=protein.residue_index,
            chain_index=protein.chain_index,
            b_factors=protein.b_factors)


def interpret_contig_expr(assembly, pdb_files):
    # iterate over chains and get the chain id
    data = []
    for chain_id, chain in enumerate(assembly):
        # iterate over segments and construct
        # segment data
        for segment in chain:
            if segment["kind"] == "free":
                data.append(make_free(segment, chain=chain_id))
            else:
                data.append(make_motif(segment, pdb_files, chain=chain_id))
    return make_sample(data)

def make_sample(make_list):
    max_length = 0
    for item in make_list:
        max_length += item["max_length"]
    def _sample():
        data = []
        for item in make_list:
            data.append(item["make"]())
        return _collate(data)
    return dict(max_length=max_length, sample=_sample)

def _collate(data):
    return {
        k: np.concatenate([
            item[k] for item in data], axis=0)
        for k in data[0]
    }

def make_free(segment, chain=0):
    if "length" in segment:
        min_length = max_length = segment["length"]
    else:
        min_length = segment["start"]
        max_length = segment["stop"]
    def _make():
        length = np.random.randint(min_length, max_length + 1)
        return dict(
            atom_positions=np.zeros((length, 5, 3), dtype=np.float32),
            atom_mask=np.zeros((length, 5), dtype=np.bool_),
            has_motif=np.zeros((length,), dtype=np.bool_),
            motif_group=np.zeros((length,), dtype=np.int32),
            aa=np.full((length,), 20, dtype=np.int32),
            chain_index=np.full((length,), chain, dtype=np.int32))
    return dict(max_length=max_length, make=_make)


def make_motif(segment, pdb_files: List[Protein], chain=0):
    protein: Protein = pdb_files[segment["pdb"]]
    source_chain = segment["chain"]
    start = segment["start"]
    stop = segment["stop"]
    segment_range = np.arange(start, stop + 1, dtype=np.int32)
    motif_group = segment["group"]
    selected_chain = protein.chain_index == np.array([source_chain])[0]
    selected_residues = (protein.residue_index[:, None] == segment_range[None, :]).any(axis=1)
    selected_motif = selected_chain * selected_residues > 0
    motif_positions = np_to_ncacocb(protein.atom_positions[selected_motif])
    motif_mask = protein.atom_mask[selected_motif][:, :5]
    motif_aa = protein.aatype[selected_motif]
    length = motif_mask.shape[0]
    def _make():
        return dict(
            atom_positions=motif_positions,
            atom_mask=motif_mask,
            has_motif=motif_mask.any(axis=-1),
            motif_group=np.full((length,), motif_group, dtype=np.int32),
            aa=motif_aa,
            chain_index=np.full((length,), chain, dtype=np.int32))
    return dict(max_length=length, make=_make)


def parse_contig_expr(data: str):
    data = data.strip().replace(" ", "").split("->")
    if len(data) == 2:
        pdbs, contig = data
        pdbs = pdbs.split(",")
    elif len(data) == 1:
        contig = data
        pdbs = []
    else:
        raise ValueError("Contig string contains multiple '->'. "
                         "Contig string should have the shape [PDB_PATH+ ->]? CONTIG_EXPR")
    return pdbs, parse_assembly(contig)

def parse_assembly(contig_string: str):
    contig_string = contig_string.strip().replace(" ", "")
    chains = contig_string.split(":")
    chain_results = []
    for chain in chains:
        segments = [parse_segment(c) for c in chain.split("/")]
        chain_results.append(segments)
    return chain_results

def parse_segment(segment: str):
    fields = segment.split(",")
    result = dict()
    if len(fields) == 1:
        result["kind"] = "free"
    else:
        result["kind"] = "motif"
    for field in fields:
        result.update(parse_field(field))
    return result

def parse_field(field):
    if not "=" in field:
        cyclic = False
        if field.startswith("c"):
            cyclic = True
            field = field[1:]
        res = [int(c) for c in field.split("-")]
        if len(res) == 1:
            return dict(length=res[0], cyclic=cyclic)
        return dict(start=res[0], stop=res[1], cyclic=cyclic)
    key, value = field.split("=")
    if key in ("group", "pdb"):
        value = int(value)
    return {key: value}
