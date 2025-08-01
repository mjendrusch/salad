"""This module implements commonly used command-line option handlers for
setting up salad scripts."""

import random
import numpy as np

import jax
import jax.numpy as jnp

from salad.modules.noise_schedule_benchmark import (
    sigma_scale_framediff, sigma_scale_cosine)

def parse_default(data, parser=float, default_key="none", default=None):
    """Parse a string with a default / off value.
    
    Args:
        data: string to be parsed.
        parser: string parser in the non-default case. Default: float.
        default_key: string used for the default case. Default: "none".
        default: default value. Default: None.
    Returns:
        Parsed value.
    """
    if data == default_key:
        return default
    return parser(data)

def parse_num_aa(data):
    """Parse a num_aa-format string containing ":"-separated chain lengths.
    
    Chain lengths can be prefixed by "c", indicating that the chain
    should be cyclised.

    E.g. 100:200:c8 indicates that three chains should be set up:
    - chain A with 100 residues
    - chain B with 200 residues
    - chain C with 8 residues, as a cyclic peptide

    Returns:
        - num_aa, the number of amino acid residues
        - resi, the residue index of shape (num_aa,)
        - chain, the chain index of shape (num_aa,)
        - is_cyclic, a boolean flag specifying if at least
            one chain is cyclic
        - cyclic_mask, a mask of cyclic chains with shape (num_aa,)
    """
    data = data.strip()
    sizes = data.split(":")
    cyclic_mask = []
    resi = []
    chain = []
    num_aa = 0
    is_cyclic = False
    for idx, size in enumerate(sizes):
        if size.startswith("c"):
            size = int(size[1:])
            cyclic = [True]
            is_cyclic = True
        else:
            size = int(size)
            cyclic = [False]
        last_resi = resi[-1] + 50 if resi else 0
        resi += [last_resi + i for i in range(size)]
        chain += size * [idx]
        cyclic_mask += size * cyclic
        num_aa += size
    resi = jnp.array(resi)
    chain = jnp.array(chain)
    cyclic_mask = jnp.array(cyclic_mask)
    return num_aa, resi, chain, is_cyclic, cyclic_mask

def parse_out_steps(data):
    """Parse a list of output step counts separated by ","."""
    data = data.strip()
    if data == "all":
        return range(1000)
    return [int(c) for c in data.split(",")]

def sigma_scale_ve(t, sigma_max=80.0, rho=7.0):
    """Noise scale function for VE diffusion."""
    sigma_min = 0.05
    time = (sigma_max ** (1 / rho) + (1 - t) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return time

def sigma_scale_edm(t):
    """Noise scale function used for training EDM models."""
    pos_mean_sigma = 1.6
    pos_std_sigma = 1.4
    return jnp.exp(pos_mean_sigma + pos_std_sigma * jax.scipy.stats.norm.ppf(
        jnp.clip(t, 0.001, 0.999)).astype(jnp.float32))

def parse_timescale(data):
    """Parse a noise scale from a string.
    
    The string can be any python expression using the time t,
    any numpy function, e.g. np.exp as well as any of the predefined
    noise schedules cosine(t), framediff(t), ve(t, max_sigma=80, rho=7.0)
    and edm(t).
    """
    data = data.strip()
    predefined = dict(
        cosine=sigma_scale_cosine,
        framediff=sigma_scale_framediff,
        ve=sigma_scale_ve,
        edm=sigma_scale_edm,
    )
    return lambda t: eval(data, None, dict(t=t, np=np, **predefined))

def cloud_std_default(num_aa):
    """Default point cloud standard deviation for VP-scaled."""
    minval = num_aa ** 0.4
    return minval + np.random.rand() * 3

def parse_cloud_std(data):
    """Parse a point cloud standard deviation function for VP-scaled.
    
    The string can be any python expression involving the number of
    amino acid residues num_aa, any numpy function, e.g. np.exp
    as well as the default standard deviation default(num_aa).
    """
    data = data.strip()
    if data == "none":
        return lambda num_aa: None
    predefined = dict(
        default=cloud_std_default
    )
    return lambda num_aa: eval(data, None, dict(num_aa=num_aa, np=np, **predefined))

def parse_dssp(data):
    """Parse a secondary structure specification.
    
    Loops are specified by "L", helices by "H" and strands by "E".
    Residues without a specification are labelled by "_" or "X".
    """
    DSSP_CODE = "LHE_X"
    if data == "none":
        return None
    if data == "random":
        return random_dssp
    result = np.array([DSSP_CODE.index(c) for c in data.strip()], dtype=np.int32)
    result = np.clip(result, 0, 3)
    return result

def random_dssp_mean(max_loop=0.5):
    """Sample a random secondary structure distribution.
    
    Args:
        max_loop: maximum frequency of loop residues.
    Returns:
        Numpy array with three entries:
        [fraction_loop, fraction_helix, fraction_strand]
    """
    helix = np.random.random()
    sheet = 1 - helix
    loop = np.random.random() * max_loop
    return np.array([loop, (1 - loop) * helix, (1 - loop) * sheet])

def random_dssp(count, p=0.5, return_string=False):
    """Sample a random secondary structure string of a fixed length.

    Args:
        count: number of residues in the secondary structure string.
        p: probability of dropping secondary structure specification. Default: 0.5
    Returns:
        Integer encoded secondary structure specification of shape (count,).
    """
    loop, helix, sheet = random_dssp_mean()
    helix = int(helix * count)
    if helix < 6:
        helix = 0
    min_helix_count = 1
    max_helix_count = helix // 6
    min_helix_count = min(min_helix_count, max_helix_count)
    if helix == 0:
        num_helices = 0
    elif min_helix_count == max_helix_count:
        num_helices = min_helix_count
    else:
        num_helices = np.random.randint(min_helix_count, max_helix_count)
    sheet = int(sheet * count)
    if sheet < 8:
        sheet = 0
    loop = count - (helix + sheet)
    min_sheet_count = 2
    max_sheet_count = sheet // 4
    if sheet == 0:
        num_sheets = 0
    elif min_sheet_count == max_sheet_count:
        num_sheets = min_sheet_count
    else:
        num_sheets = np.random.randint(min_sheet_count, max_sheet_count)
    helices = [6 for _ in range(num_helices)] 
    sheets = [4 for _ in range(num_sheets)]
    loops = [0 for _ in range(num_helices + num_sheets + 1)]
    while sum(sheets) < sheet:
        index = np.random.randint(num_sheets)
        sheets[index] += 1
    while sum(helices) < helix:
        index = np.random.randint(num_helices)
        helices[index] += 1
    while sum(loops) < loop:
        index = np.random.randint(0, len(loops))
        loops[index] += 1
    helices = ["_" + "H" * (num - 2) + "_" if random.random() > p else "_" * num for num in helices]
    sheets = ["_" + "E" * (num - 2) + "_" if random.random() > p else "_" * num for num in sheets]
    loops = ["L" * num if random.random() < p else "_" * num for num in loops]
    structured = helices + sheets
    random.shuffle(structured)
    dssp_string = loops[0] + "".join([s + l for s, l in zip(structured, loops[1:])])
    dssp = parse_dssp(dssp_string)
    if return_string:
        return dssp, dssp_string
    return dssp
