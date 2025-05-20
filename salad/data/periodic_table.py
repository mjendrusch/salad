"""This module contains constants for element types.

Not used in the manuscript.
"""

import numpy as np

# element names in the periodic table
PT_TABLE = [
    ["H"] + [""] * 30 + ["He"],
    ["Li", "Be"] + [""] * 24 + ["B", "C", "N", "O", "F", "Ne"],
    ["Na", "Mg"] + [""] * 24 + ["Al", "Si", "P", "S", "Cl", "Ar"],
    ["K", "Ca", "Sc"] + [""] * 14 + ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
    ["Rb", "Sr", "Y"] + [""] * 14 + ["Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
    ["Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
    ["Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs"] + 10 * [""]
]
PT_TABLE = np.array(PT_TABLE)
# linear order of elements
ATOM_TYPE_ORDER = PT_TABLE.reshape(-1)
ATOM_TYPE_ORDER = ATOM_TYPE_ORDER[ATOM_TYPE_ORDER != ""]
def pt_at(x):
    """Finds the row and column of an element symbol 'x' in the periodic table array."""
    match = PT_TABLE == x
    col = np.argmax(match.any(axis=0))
    row = np.argmax(match.any(axis=1))
    return row, col

def make_periodic_table():
    r"""Constructs atomic number and orbital information for the periodic table."""
    pt = np.array(PT_TABLE)
    has_entry = np.vectorize(lambda x: 1 if x != "" else 0)(pt)
    last_shell = np.where(has_entry, np.cumsum(has_entry, axis=1), 0)
    protons = np.where(has_entry, np.cumsum(has_entry).reshape(*pt.shape), 0)
    sorb = np.where(has_entry, np.clip(np.cumsum(has_entry, axis=1), 0, 2), 0)
    for elem in ("Cr", "Cu", "Nb", "Ru", "Rh", "Ag", "Au", "Pt"):
        sorb[pt_at(elem)] = 1
    for elem in ("Pd",):
        sorb[pt_at(elem)] = 0
    forb = 14 * np.ones_like(has_entry)
    forb[:5] = 0
    forb[5:, :3] = 0
    forb[pt_at("Th")] = 0
    forb[pt_at("Ce")] = 1
    forb[pt_at("Pa")] = 2
    for elem in ("Pr", "U"):
        forb[pt_at(elem)] = 3
    for elem in ("Nd", "Np"):
        forb[pt_at(elem)] = 4
    forb[pt_at("Pm")] = 5
    for elem in ("Sm", "Pu"):
        forb[pt_at(elem)] = 6
    for elem in ("Eu", "Gd", "Am", "Cm"):
        forb[pt_at(elem)] = 7
    for elem in ("Tb", "Bk"):
        forb[pt_at(elem)] = 9
    for elem in ("Cf", "Dy"):
        forb[pt_at(elem)] = 10
    for elem in ("Ho", "Es"):
        forb[pt_at(elem)] = 11
    for elem in ("Er", "Fm"):
        forb[pt_at(elem)] = 12
    for elem in ("Tm", "Md"):
        forb[pt_at(elem)] = 13
    forb *= has_entry
    fills_p = np.zeros_like(has_entry)
    fills_p[1:, -6:] = 1
    fills_p *= has_entry
    porb = np.where(has_entry, np.clip(np.cumsum(fills_p, axis=1), 0, 6), 0)
    dorb = last_shell - porb - sorb - forb
    orbs = np.stack((sorb, porb, dorb, forb), axis=-1)
    valence = sorb + porb
    return dict(
        names=PT_TABLE,
        mask=has_entry > 0,
        protons=protons,
        orbs=orbs,
        valence=valence
    )
periodic_table = make_periodic_table()

def index_atoms(atom_types):
    return np.argmax(atom_types[:, None] == ATOM_TYPE_ORDER, axis=1)

def apply_pt(atom_index):
    return {
        key: value[periodic_table["mask"]].reshape(-1, *value.shape[2:])[atom_index]
        for key, value in periodic_table.items()
    }
