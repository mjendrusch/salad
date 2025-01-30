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

import jax
import jax.numpy as jnp

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

def _unfold(a: jnp.ndarray, window: int, axis: int):
    idx = jnp.arange(window)[:, None] + jnp.arange(a.shape[axis] - window + 1)[None, :]
    unfolded = jnp.take(a, idx, axis=axis)
    return  jnp.moveaxis(unfolded, axis - 1, -1)

def _check_input(coord):
    org_shape = coord.shape
    assert (len(org_shape)==3), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    return coord, org_shape

def _get_peptide_bond_h_position(coord: jnp.ndarray) -> jnp.ndarray:
    vec_cn = coord[1:, 0] - coord[:-1, 2]
    vec_cn = vec_cn / jnp.maximum(jnp.linalg.norm(vec_cn, axis=-1, keepdims=True), 1e-3)
    vec_can = coord[1:, 0] - coord[1:, 1]
    vec_can = vec_can / jnp.maximum(jnp.linalg.norm(vec_can, axis=-1, keepdims=True), 1e-3)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / jnp.maximum(jnp.linalg.norm(vec_nh, axis=-1, keepdims=True), 1e-3)
    return coord[1:, 0] + 1.01 * vec_nh

def get_hbond_map(
    coord: jnp.ndarray,
    mask: jnp.ndarray,
    cutoff: float=DEFAULT_CUTOFF,
    margin: float=DEFAULT_MARGIN,
    ) -> jnp.ndarray:
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
    d_on = jnp.linalg.norm(omap - nmap, axis=-1) + 1e-3
    d_ch = jnp.linalg.norm(cmap - hmap, axis=-1) + 1e-3
    d_oh = jnp.linalg.norm(omap - hmap, axis=-1) + 1e-3
    d_cn = jnp.linalg.norm(cmap - nmap, axis=-1) + 1e-3
    # electrostatic interaction energy
    e = jnp.pad(CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn)*CONST_F, [[1, 0], [0, 1]])
    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~jnp.eye(num_aa, dtype=bool)
    local_mask *= ~jnp.diag(jnp.ones(num_aa - 1, dtype=bool), k=-1)
    local_mask *= ~jnp.diag(jnp.ones(num_aa - 2, dtype=bool), k=-2)
    # hydrogen bond map (continuous value extension of original definition)
    hbond_map = jnp.clip(cutoff - margin - e, a_min=-margin, a_max=margin)
    hbond_map = (jnp.sin(hbond_map / margin * jnp.pi / 2) + 1.0) / 2
    hbond_map = jnp.where(local_mask, hbond_map, 0)
    # return h-bond map
    hbond_map = jnp.where(mask, hbond_map, 0)
    return hbond_map

def assign_dssp(coord: jnp.ndarray,
                batch_index: jnp.ndarray,
                mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    print(coord.shape)
    if coord.shape[0] < 5:
        # if coordinates are too short (< 3) for DSSP computation,
        # just return unknown DSSP (index = 3)
        dssp = 3 * jnp.ones(batch_index.shape, dtype=jnp.int32)
        blocks = jnp.zeros_like(batch_index, dtype=jnp.int32)
        block_adjacency = jnp.zeros((
            batch_index.shape[0], batch_index.shape[0]),
            dtype=jnp.bool_)
        return dssp, blocks, block_adjacency
    # check input
    coord, _ = _check_input(coord)
    # create pair mask
    pair_mask = batch_index[:, None] == batch_index[None, :]
    pair_mask *= mask[:, None] * mask[None, :]
    # get hydrogen bond map
    hbmap = get_hbond_map(coord, pair_mask)
    hbmap = jnp.swapaxes(hbmap, -1, -2)
    # identify turn 3, 4, 5
    turn3 = jnp.diagonal(hbmap, axis1=-2, axis2=-1, offset=3) > 0.
    turn4 = jnp.diagonal(hbmap, axis1=-2, axis2=-1, offset=4) > 0.
    turn5 = jnp.diagonal(hbmap, axis1=-2, axis2=-1, offset=5) > 0.
    # assignment of helical sses
    h3 = jnp.pad(turn3[:-1] * turn3[1:], [[1, 3]])
    h4 = jnp.pad(turn4[:-1] * turn4[1:], [[1, 4]])
    h5 = jnp.pad(turn5[:-1] * turn5[1:], [[1, 5]])
    # helix4 first
    helix4 = h4 + jnp.roll(h4, 1, 0) + jnp.roll(h4, 2, 0) + jnp.roll(h4, 3, 0)
    h3 = h3 * ~jnp.roll(helix4, -1, 0) * ~helix4 # helix4 is higher prioritized
    h5 = h5 * ~jnp.roll(helix4, -1, 0) * ~helix4 # helix4 is higher prioritized
    helix3 = h3 + jnp.roll(h3, 1, 0) + jnp.roll(h3, 2, 0)
    helix5 = h5 + jnp.roll(h5, 1, 0) + jnp.roll(h5, 2, 0) + jnp.roll(h5, 3, 0) + jnp.roll(h5, 4, 0)
    # identify bridge
    unfoldmap = _unfold(_unfold(hbmap, 3, -2), 3, -2) > 0.
    unfoldmap_rev = jnp.swapaxes(unfoldmap, 0, 1)
    p_bridge = (unfoldmap[:, :, 0, 1] * unfoldmap_rev[:, :, 1, 2]) + (unfoldmap_rev[:, :, 0, 1] * unfoldmap[:, :, 1, 2])
    p_bridge = jnp.pad(p_bridge, [[1, 1], [1, 1]])
    p_bridge = jnp.where(pair_mask, p_bridge, 0)
    a_bridge = (unfoldmap[:, :, 1, 1] * unfoldmap_rev[:, :, 1, 1]) + (unfoldmap[:, :, 0, 2] * unfoldmap_rev[:, :, 0, 2])
    a_bridge = jnp.pad(a_bridge, [[1, 1], [1, 1]])
    a_bridge = jnp.where(pair_mask, a_bridge, 0)
    # ladder
    ladder = (p_bridge + a_bridge).sum(-1) > 0
    # strand
    p_res = p_bridge.sum(axis=-1) > 0
    p_strand = p_res * jnp.roll(p_res, -1, 0) + p_res * jnp.roll(p_res, 1, 0) > 0
    a_res = a_bridge.sum(axis=-1) > 0
    a_strand = a_res * jnp.roll(a_res, -1, 0) + a_res * jnp.roll(a_res, 1, 0) > 0
    # all strands
    strand = a_strand + p_strand + a_res + p_res > 0
    # H, E, L of C3
    helix = (helix3 + helix4 + helix5) > 0
    strand = ladder
    loop = (~helix * ~strand)
    index = jnp.argmax(jnp.stack([loop, helix, strand], axis=-1), axis=-1)
    def block_iter(carry, data):
        state, block_index = carry
        new_state = data
        block_index = jnp.where(new_state != state, block_index + 1, block_index)
        return (new_state, block_index), block_index
    _, blocks = jax.lax.scan(block_iter, (index[0], jnp.zeros((), dtype=jnp.int32)), index)
    same_block = blocks[:, None] == blocks[None, :]
    sheet_sheet = (index[:, None] == 2) * (index[None, :] == 2)
    distance = jnp.linalg.norm(coord[:, None, CA_INDEX] - coord[None, :, CA_INDEX], axis=-1)
    block_distance = (1e6 * jnp.ones_like(distance)).at[blocks[:, None], blocks[None, :]].min(distance)[blocks[:, None], blocks[None, :]]
    block_adjacency = jnp.where(sheet_sheet, block_distance <= SHEET_ADJACENCY, block_distance <= OTHER_ADJACENCY)
    block_adjacency = jnp.where(same_block + loop[:, None] + loop[None, :] > 0, 0, block_adjacency)
    secondary_structure = index
    return secondary_structure, blocks, block_adjacency

def drop_dssp(key, secondary_structure, blocks, block_adjacency, p_drop=0.2):
    drop_mask = jax.random.bernoulli(key, p_drop, secondary_structure.shape)[blocks]
    drop_pair = drop_mask[:, None] + drop_mask[None, :] > 0
    secondary_structure = jnp.where(drop_mask, 0, secondary_structure)
    block_adjacency = jnp.where(drop_pair, 0, block_adjacency)
    return secondary_structure, block_adjacency
