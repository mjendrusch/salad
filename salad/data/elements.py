import numpy as np

# Adapted from PyDSSP

# MIT License

# Copyright (c) 2022 Shintaro Minami

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Tuple

CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0
SHEET_ADJACENCY = 8.0
OTHER_ADJACENCY = 8.0
N_INDEX = 0
CA_INDEX = 1
CO_INDEX = 2
O_INDEX = 3

def _unfold(a: np.ndarray, window: int, axis: int):
    idx = np.arange(window)[:, None] + np.arange(a.shape[axis] - window + 1)[None, :]
    unfolded = np.take(a, idx, axis=axis)
    return  np.moveaxis(unfolded, axis - 1, -1)

def _check_input(coord):
    org_shape = coord.shape
    assert (len(org_shape)==3), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    return coord, org_shape

def _get_peptide_bond_h_position(coord: np.ndarray) -> np.ndarray:
    vec_cn = coord[1:, 0] - coord[:-1, 2]
    vec_cn = vec_cn / np.maximum(np.linalg.norm(vec_cn, axis=-1, keepdims=True), 1e-3)
    vec_can = coord[1:, 0] - coord[1:, 1]
    vec_can = vec_can / np.maximum(np.linalg.norm(vec_can, axis=-1, keepdims=True), 1e-3)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / np.maximum(np.linalg.norm(vec_nh, axis=-1, keepdims=True), 1e-3)
    return coord[1:, 0] + 1.01 * vec_nh

def np_get_hbond_map(
    coord: np.ndarray,
    mask: np.ndarray,
    cutoff: float=DEFAULT_CUTOFF,
    margin: float=DEFAULT_MARGIN,
    ) -> np.ndarray:
    # check input
    coord, _ = _check_input(coord)
    num_aa, num_atoms, _ = coord.shape
    # add pseudo-H atom
    assert (num_atoms >= 4), "Number of atoms should be at least 4 (N,CA,C,O)"
    coord = coord[:, :4]
    h = _get_peptide_bond_h_position(coord)
    # distance matrix
    nmap = coord[1:, None, N_INDEX]
    hmap = h[:, None]
    cmap = coord[None, :-1, CO_INDEX]
    omap = coord[None, :-1, O_INDEX]
    d_on = np.linalg.norm(omap - nmap, axis=-1) + 1e-3
    d_ch = np.linalg.norm(cmap - hmap, axis=-1) + 1e-3
    d_oh = np.linalg.norm(omap - hmap, axis=-1) + 1e-3
    d_cn = np.linalg.norm(cmap - nmap, axis=-1) + 1e-3
    # electrostatic interaction energy
    e = np.pad(CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn)*CONST_F, [[1, 0], [0, 1]])
    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~np.eye(num_aa, dtype=bool)
    local_mask *= ~np.diag(np.ones(num_aa - 1, dtype=bool), k=-1)
    local_mask *= ~np.diag(np.ones(num_aa - 2, dtype=bool), k=-2)
    # hydrogen bond map (continuous value extension of original definition)
    hbond_map = np.clip(cutoff - margin - e, a_min=-margin, a_max=margin)
    hbond_map = (np.sin(hbond_map / margin * np.pi / 2) + 1.0) / 2
    hbond_map = np.where(local_mask, hbond_map, 0)
    # return h-bond map
    hbond_map = np.where(mask, hbond_map, 0)
    return hbond_map

def np_assign_dssp(coord: np.ndarray,
                   batch_index: np.ndarray,
                   mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coord.shape[0] < 5:
        # if coordinates are too short (< 3) for DSSP computation,
        # just return unknown DSSP (index = 3)
        dssp = 3 * np.ones(batch_index.shape, dtype=np.int32)
        blocks = np.zeros_like(batch_index, dtype=np.int32)
        block_adjacency = np.zeros((
            batch_index.shape[0], batch_index.shape[0]),
            dtype=np.bool_)
        return dssp, blocks, block_adjacency
    # check input
    coord, _ = _check_input(coord)
    # create pair mask
    pair_mask = batch_index[:, None] == batch_index[None, :]
    pair_mask *= mask[:, None] * mask[None, :]
    # get hydrogen bond map
    hbmap = np_get_hbond_map(coord, pair_mask)
    hbmap = np.swapaxes(hbmap, -1, -2)
    # identify turn 3, 4, 5
    turn3 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=3) > 0.
    turn4 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=4) > 0.
    turn5 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=5) > 0.
    # assignment of helical sses
    h3 = np.pad(turn3[:-1] * turn3[1:], [[1, 3]])
    h4 = np.pad(turn4[:-1] * turn4[1:], [[1, 4]])
    h5 = np.pad(turn5[:-1] * turn5[1:], [[1, 5]])
    # helix4 first
    helix4 = h4 + np.roll(h4, 1, 0) + np.roll(h4, 2, 0) + np.roll(h4, 3, 0)
    h3 = h3 * ~np.roll(helix4, -1, 0) * ~helix4 # helix4 is higher prioritized
    h5 = h5 * ~np.roll(helix4, -1, 0) * ~helix4 # helix4 is higher prioritized
    helix3 = h3 + np.roll(h3, 1, 0) + np.roll(h3, 2, 0)
    helix5 = h5 + np.roll(h5, 1, 0) + np.roll(h5, 2, 0) + np.roll(h5, 3, 0) + np.roll(h5, 4, 0)
    # identify bridge
    unfoldmap = _unfold(_unfold(hbmap, 3, -2), 3, -2) > 0.
    unfoldmap_rev = np.swapaxes(unfoldmap, 0, 1)
    p_bridge = (unfoldmap[:, :, 0, 1] * unfoldmap_rev[:, :, 1, 2]) + (unfoldmap_rev[:, :, 0, 1] * unfoldmap[:, :, 1, 2])
    p_bridge = np.pad(p_bridge, [[1, 1], [1, 1]])
    p_bridge = np.where(pair_mask, p_bridge, 0)
    a_bridge = (unfoldmap[:, :, 1, 1] * unfoldmap_rev[:, :, 1, 1]) + (unfoldmap[:, :, 0, 2] * unfoldmap_rev[:, :, 0, 2])
    a_bridge = np.pad(a_bridge, [[1, 1], [1, 1]])
    a_bridge = np.where(pair_mask, a_bridge, 0)
    # ladder
    ladder = (p_bridge + a_bridge).sum(-1) > 0
    # strand
    p_res = p_bridge.sum(axis=-1) > 0
    p_strand = p_res * np.roll(p_res, -1, 0) + p_res * np.roll(p_res, 1, 0) > 0
    a_res = a_bridge.sum(axis=-1) > 0
    a_strand = a_res * np.roll(a_res, -1, 0) + a_res * np.roll(a_res, 1, 0) > 0
    # all strands
    strand = a_strand + p_strand + a_res + p_res > 0
    # H, E, L of C3
    helix = (helix3 + helix4 + helix5) > 0
    strand = ladder
    loop = (~helix * ~strand)
    index = np.argmax(np.stack([loop, helix, strand], axis=-1), axis=-1)
    secondary_structure = index
    return secondary_structure
# adapted from pyDSSP up to here.

def fragment_object(positions, mask, center_ratio=20, num_kmeans_iter=5):
    num_object = mask.sum()
    if num_object <= 0:
        return []
    num_centers = num_object // center_ratio + 1
    nn_pos = positions
    # k means
    centers = []
    centers.append(nn_pos[np.random.randint(nn_pos.shape[0])])
    for i in range(num_centers - 1):
        cc = np.array(centers)
        next_center = np.argmax(np.linalg.norm(nn_pos[:, None] - cc[None, :], axis=-1).min(axis=1))
        centers.append(nn_pos[next_center])
    centers = np.array(centers)
    for idx in range(num_kmeans_iter):
        dist = np.linalg.norm(centers[:, None] - nn_pos[None, :], axis=-1)
        assignment = dist.argmin(axis=0)
        center_count = np.zeros_like(centers[:, 0])
        np.add.at(center_count, assignment, 1)
        center_val = np.zeros_like(centers)
        np.add.at(center_val, assignment, nn_pos)
        centers = center_val / np.maximum(center_count[..., None], 1)
    dist = np.linalg.norm(centers[:, None] - nn_pos[None, :], axis=-1)
    assignment = dist.argmin(axis=0)
    nn_elements = []
    for i in range(num_centers):
        assigned = nn_pos[assignment == i]
        center = assigned.mean(axis=0)
        dirs = assigned - center
        svd = np.linalg.svd((dirs[:, :, None] * dirs[:, None, :]).mean(axis=0))
        long = svd.Vh[0]
        dirs_sorted_index = np.argsort((dirs * long).sum(axis=-1), axis=0)
        start = assigned[dirs_sorted_index[:3]].mean(axis=0)
        end = assigned[dirs_sorted_index[-3:]].mean(axis=0)
        nn_elements.append((start, center, end))
    return nn_elements

def to_elements(data, num_kmeans_iter=5, assignment=np_assign_dssp):
    is_smol = data["residue_type"] == "SMOL"
    is_na = data["residue_type"] == "DNA"
    is_na += data["residue_type"] == "RNA"
    is_na = is_na > 0
    dssp = assignment(data["position"], data["chain_index"], data["atom_mask"].any(axis=1))
    is_loop = dssp == 0
    dssp = dssp[~is_loop]
    blocks = blocks[~is_loop]
    ca = data["position"][~is_loop][:, 1]
    nn = data["position"][is_na]
    smol_pos = data["position"][is_smol][:, 0]
    block_names = np.unique(blocks)
    block_num_mean = (4, 2)
    dssps = []
    coords = []
    for block_name in block_names:
        selected = blocks == block_name
        block_dssp = dssp[selected][0]
        block_dssp -= 1
        if block_dssp == 1 and len(selected) < 2:
            continue
        block_ca = ca[selected]
        size = block_ca.shape[0]
        center = block_ca[max(size//2 - 2, 0):min(size//2 + 2, size)].mean(axis=0)
        start = block_ca[:block_num_mean[block_dssp]].mean(axis=0)
        end = block_ca[-block_num_mean[block_dssp]:].mean(axis=0)
        coords.append((start, center, end))
        dssps.append(block_dssp)
    # assign nucleic acids
    nn_pos = nn[:, 11]
    nn_elements = fragment_object(nn_pos, is_na, center_ratio=20, num_kmeans_iter=num_kmeans_iter)
    coords += nn_elements
    dssps += len(nn_elements) * [2] # 2 = NA
    # assign small molecules
    smol_pos = smol_pos
    smol_elements = fragment_object(smol_pos, is_smol, center_ratio=20, num_kmeans_iter=num_kmeans_iter)
    coords += smol_elements
    dssps += len(smol_elements) * [3] # 3 = SMOL
    return dict(dssp=np.array(dssps, dtype=np.int32), position=np.array(coords, dtype=np.float32))
