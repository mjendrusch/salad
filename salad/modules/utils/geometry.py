r"""This module contains utility functions for working with sparse
representations of protein geometry."""

from typing import Optional, Union, Tuple, Iterable, Any, Dict

import numpy as np

import jax
import jax.numpy as jnp

import haiku as hk

from salad.aflib.common import residue_constants
from salad.aflib.model.all_atom_multimer import (
    make_transform_from_reference, torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos)
from salad.aflib.model.geometry import Vec3Array, Rigid3Array

def make_backbone_affine(
    positions: Vec3Array,
    mask: jnp.ndarray,
    atoms: Optional[Iterable[str]] = None,
    atom_order: Optional[Dict[str, int]] = None
    ) -> Tuple[Rigid3Array, jnp.ndarray]:
    """Make backbone Rigid3Array and mask.
    
    Args:
        positions: atom positions of shape (N, 3+, 3).
        mask: atom mask.
        atoms: list of 3 atom names to use for frame construction. Default: N, CA, C
        atom_order: order of atom names in positions. Default: atom14.
    Returns:
        Rigid3Array of residue frames.
    """
    if atom_order is None:
        atom_order = residue_constants.atom_order
    if atoms is None:
        atoms = ('N', 'CA', 'C')
    a, b, c = [residue_constants.atom_order[name] for name in atoms]

    rigid_mask = (mask[..., a] * mask[..., b] * mask[..., c]).astype(
        jnp.float32)

    rigid = make_transform_from_reference(
        a_xyz=positions[..., a],
        b_xyz=positions[..., b],
        c_xyz=positions[..., c])

    return rigid, rigid_mask

def extract_aa_frames(positions: Vec3Array) -> Tuple[Rigid3Array, Vec3Array]:
    """Extract frames from protein backbone positions.
    
    Args:
        positions: Vec3Array of amino acid backbone atoms in atom14 format.
    Returns:
        Rigid3Array of residue frames and Vec3Array of local-frame
        side chain atom positions.
    """
    rigids, _ = make_backbone_affine(positions, jnp.ones((positions.shape[0], 14)), None)
    local_positions = rigids[..., None].apply_inverse_to_point(positions)
    return rigids, local_positions

def extract_na_frames(positions: Vec3Array):
    """Extract frames from nucleic acid backbone positions using O4, C1 and C2.

    Args:
        positions: Vec3Array of nucleic acid backbone atoms in atom14 format.
    Returns:
        Rigid3Array of residue frames and Vec3Array of local-frame
        side chain atom positions.
    """
    rigids, _ = make_backbone_affine(positions, atoms=('O4', 'C1', 'C2'))
    local_positions = rigids[..., None].apply_inverse_to_point(positions)
    return rigids, local_positions

def extract_aa_relmap(positions: Vec3Array, atom_mask: jnp.ndarray):
    """Extract relative atom positions between residue pairs.
    
    Args:
        positions: Vec3Array of atom14 format atom positions.
        atom_mask: atom14 format atom mask.
    Returns:
        Relative position map of shape (N, N, 3+, 3) and corresponding
        mask of shape (N, N, 3+).
    """
    frames, _ = extract_aa_frames(positions)
    relmap = frames[:, None, None].apply_inverse_to_point(positions[None])
    rel_mask = atom_mask[:, None, 1:2] * atom_mask[None, :]
    return relmap, rel_mask

def extract_pseudo_distmap(positions: Vec3Array, resi, chain, batch, mask: jnp.ndarray,
                           iterations=2):
    """DEPRECATED."""
    same_batch = batch[:, None] == batch[None, :]
    same_chain = same_batch * (chain[:, None] == chain[None, :])
    pair_mask = mask[:, None] * mask[None, :] * same_batch
    dist = (positions[:, None] - positions[None, :]).norm()
    dist = jnp.where(pair_mask, dist, jnp.inf)
    # maximum linear chain distance between two AAs
    pseudo_dist = abs(resi[:, None] - resi[None, :]) * 4
    pseudo_dist = jnp.where(same_chain, pseudo_dist, jnp.inf)
    # combine with ground-truth distances
    pseudo_dist = jnp.minimum(pseudo_dist, dist)
    for idx in range(iterations):
        pseudo_dist = (pseudo_dist[:, :, None] + pseudo_dist[None, :, :]).min(axis=1)
    return pseudo_dist

def bond_angle(x, y, z):
    """Compute the bond angle between three atoms x, y and z.
    
    Args:
        x, y, z: atom positions of shape (..., 3).
    Returns:
        Bond angle with y as the central atom.
    """
    left = x - y
    right = z - y
    cos_tau = (left * right).sum(axis=-1) / jnp.maximum(jnp.linalg.norm(left, axis=-1) * jnp.linalg.norm(right, axis=-1), 1e-6)
    return jnp.arccos(cos_tau) / jnp.pi * 180

def dihedral_angle(a, b, c, d):
    """Compute the dihedral angle for four atoms a, b, c, d.
    
    Args:
        a, b, c, d: atom positions of shape (..., 3).
    Returns:
        Dihedral angle along a, b, c and d.
    """
    x = b - a
    y = c - b
    z = d - c
    y_norm = jnp.linalg.norm(y, axis=-1)
    result = jnp.arctan2(y_norm * (x * jnp.cross(y, z)).sum(axis=-1),
                         (jnp.cross(x, y) * jnp.cross(y, z)).sum(axis=-1))
    return result / jnp.pi * 180

def single_protein_sidechains(aatype: jnp.ndarray, frames: Rigid3Array, angles: jnp.ndarray):
    """Compute side chain atom positions given backbone frames and dihedral angles.
    
    Args:
        aatype: integer amino acid type (0-19) of shape (N,).
        frames: amino acid backbone frames of shape (N,).
        angles: side chain dihedral angles of shape (N, 7, 2).
    Returns:
        atom14 format all-atom positions of shape (N, 14, 3).
    """
    # Map torsion angles to frames.
    # geometry.Rigid3Array with shape (N, 8)
    all_frames_to_global = torsion_angles_to_frames(
        aatype,
        frames,
        angles
    )

    # Use frames and literature positions to create the final atom coordinates.
    # geometry.Vec3Array with shape (N, 14)
    pred_positions = frames_and_literature_positions_to_atom14_pos(
        aatype, all_frames_to_global
    )

    return pred_positions, all_frames_to_global

def sequence_relative_position(count: Optional[int] = 32,
                               one_hot=False,
                               cyclic=False,
                               identify_ends=False,
                               pseudo_chains=False):
    """Compute sequence relative positions features for a protein chain.
    
    Args:
        count: returns separate features for signed distances from -count to +count.
        one_hot: return one-hot encoded features. Default: False.
        cyclic: cyclise one or more chains. Default: False.
        identify_ends: use the same representation for +count and -count. Default: False.
        pseudo_chains: represent distances across chains by +count or -count
            instead of a separate label. Default: False.
    Returns:
        A function computing relative position features given
        residue, chain and batch indices (N,), as well a neighbour array (N, K)
        and optionally a cyclic_mask (N,) which specifies which chains should
        be cyclised.
    """
    def inner(resi, chain, batch, neighbours=None, cyclic_mask=None):
        compare_index = (None, slice(None))
        if neighbours is not None:
            compare_index = neighbours
        same_chain = chain[:, None] == chain[compare_index]
        same_batch = batch[:, None] == batch[compare_index]
        dist = resi[:, None] - resi[compare_index]
        flat_resi = jnp.arange(resi.shape[0], dtype=jnp.int32)
        if cyclic:
            lengths = index_count(chain, jnp.ones_like(chain, dtype=jnp.bool_))
            wrap = abs(dist) > lengths[:, None] / 2
            # control cyclic wrapping per residue/chain
            if cyclic_mask is not None:
                wrap = wrap * cyclic_mask[:, None]
            dist = jnp.where(
                wrap,
                jnp.where(dist < 0,
                         dist % lengths[:, None],
                         dist % lengths[:, None] - lengths[:, None]),
                dist)
        dist = jnp.clip(dist, -count, count) + count
        if identify_ends:
            count_total = 2 * count - 2
            dist = jnp.where(dist == 0, 2 * count - 2, dist - 1)
            dist = jnp.where(same_chain, dist, 2 * count - 2)
            dist = jnp.where(same_batch, dist, 2 * count - 2)
        elif pseudo_chains:
            flat_dist = flat_resi[:, None] - flat_resi[compare_index]
            flat_dist = jnp.where(flat_dist >= 0, 0, 2 * count - 1)
            count_total = 2 * count + 2
            dist = jnp.where(same_chain, dist, flat_dist)
            dist = jnp.where(same_batch, dist, 2 * count + 1)
        else:
            count_total = 2 * count + 2
            dist = jnp.where(same_chain, dist, 2 * count + 1)
            dist = jnp.where(same_batch, dist, 2 * count + 1)
        if one_hot:
            dist = jax.nn.one_hot(dist, count_total, axis=-1)
        return dist
    return inner

def get_neighbours(count: int):
    """Select `count` nearest neighbours per residue.
    
    Args:
        count: number of neighbours to select.
    Returns:
        A function selecting neighbours given a distance matrix (N, N)
        a residue mask (N,) and an optional neighbour array (N, K).
        The result contains the next `count` nearest neighbours for
        each residue concatenated with the input neighbour array.
    """
    def inner(distance: jnp.ndarray,
              mask: jnp.ndarray,
              neighbours: Optional[jnp.ndarray] = None):
        index = jnp.arange(0, distance.shape[0])
        distance = jnp.where(mask, distance, jnp.inf)
        # if a neighbour array was passed, make the distance of neighbours
        # in that array infinite. This way neighbours will not be selected
        # more than once.
        if neighbours is not None:
            update = jnp.where(neighbours != -1, jnp.inf, distance[index[:, None], neighbours])
            distance = distance.at[index[:, None], neighbours].set(update)
        # select the `count` nearest neighbours
        knn = jnp.argsort(distance, axis=-1)[..., :count]
        knn = jnp.where(distance[index[:, None], knn] < jnp.inf, knn, -1)
        # and concantenate with the input neighbour array
        if neighbours is not None:
            knn = jnp.concatenate((neighbours, knn), axis=-1)
        return knn
    return inner

def extract_neighbours(num_index=16, num_spatial=16, num_random=16):
    """Extracts the default set of nearest neighbours of each residue.
    
    Args:
        num_index: number of neighbours using residue index distance.
        num_spatial: number of nearest neighbours using euclidean distance d.
        num_random: number of neighbours sampled with probability 1 / d^3.

    Returns:
        A function extracting per-residue nearest neighbours given
        atom positions (N, 3+, 3), residue, chain and batch index
        and a residue mask.
    """
    def inner(pos, resi, chain, item, mask):
        neighbours = get_index_neighbours(num_index)(resi, chain, item, mask)
        neighbours = get_spatial_neighbours(num_spatial)(pos[:, 1], item, mask, neighbours)
        neighbours = get_random_neighbours(num_random)(pos[:, 1], item, mask, neighbours)
        return neighbours
    return inner

def get_index_neighbours(count: int):
    """Extracts the `count` nearest neighbours based on residue index."""
    def inner(resi, chain, item, mask, neighbours=None):
        distance = abs(resi[:, None] - resi[None, :])
        same_chain = chain[:, None] == chain[None, :]
        same_item = item[:, None] == item[None, :]
        mask = same_item * same_chain * (mask[:, None] * mask[None, :])
        return get_neighbours(count)(distance, mask, neighbours)
    return inner

def get_spatial_neighbours(count: int):
    """Extracts the `count` nearest neighbours based on euclidean distance."""
    def inner(pos: Vec3Array | jax.Array, item, mask, neighbours=None):
        if not isinstance(pos, Vec3Array):
            pos = Vec3Array.from_array(pos)
        distance = (pos[:, None] - pos[None, :]).norm()
        same_item = item[:, None] == item[None, :]
        distance = jnp.where(same_item, distance, jnp.inf)
        mask = (mask[:, None] * mask[None, :] * same_item)
        return get_neighbours(count)(distance, mask, neighbours)
    return inner

def get_random_neighbours(count: int):
    """Extracts `count` neighbours with probability 1 / d^3."""
    def inner(pos: Any, item, mask, neighbours=None):
        distance = None
        if isinstance(pos, jnp.ndarray):
            assert(pos.ndim == 2)
            if pos.shape[0] == pos.shape[1]:
                distance = pos
            else:
                pos = Vec3Array.from_array(pos)
        if distance is None:
            distance = (pos[:, None] - pos[None, :]).norm()
        same_item = item[:, None] == item[None, :]
        # apply gumbel topk trick to select random neighbours
        weight = -3 * jnp.log(distance + 1e-6)
        uniform = jax.random.uniform(hk.next_rng_key(), weight.shape, dtype=weight.dtype, minval=1e-6, maxval=1 - 1e-6)
        gumbel = jnp.log(-jnp.log(uniform))
        weight = weight - gumbel
        distance = -weight
        distance = jnp.where(same_item, distance, jnp.inf)
        mask = (mask[:, None] * mask[None, :] * same_item)
        return get_neighbours(count)(distance, mask, neighbours)
    return inner

def get_random_core_residues(count: int):
    """DEPRECATED."""
    def inner(item, mask):
        weight = jnp.maximum(index_count(item, mask), 1)
        weight = -jnp.log(weight)
        gumbel = jax.random.gumbel(hk.next_rng_key(), weight.shape, dtype=weight.dtype)
        weight = weight - gumbel
        sort_weight = -weight
        sort_weight = jnp.where(mask, sort_weight, jnp.inf)
        core_index = jnp.argsort(sort_weight)[:count]
        core_index = jnp.where(mask[core_index], core_index, -1)
        return core_index
    return inner

def get_contact_neighbours(count):
    """Extracts `count` neighbours with non-zero pair conditioning information."""
    def inner(pair_condition, mask, neighbours):
        # get pairs where at least one condition is True
        is_conditioned = pair_condition.any(axis=-1)
        # construct a distance matrix with random values for conditioned positions
        # and infinite distance everywhere else.
        distance = jnp.where(is_conditioned,
                             jax.random.uniform(hk.next_rng_key(), is_conditioned.shape),
                             jnp.inf)
        # get new neighbours using that distance matrix.
        # infinite distance results in a pair being masked (set to -1 in neighbours).
        # this way effectively /no additional neighbours are used when not conditioning/
        return get_neighbours(count)(distance, mask, neighbours)
    return inner

def freq_embedding(t, size=128, min_freq=0.01, max_freq=2**8):
    """Non-equivariant frequency positional embedding."""
    N_freqs = size // 2
    freqs = 2.0 ** jnp.linspace(jnp.log2(min_freq), jnp.log2(max_freq), N_freqs)
    result = t[..., None] * freqs
    return jnp.stack((jnp.sin(result), jnp.cos(result)), axis=-1).reshape(t.shape[0], -1)

def gaussian_rbf(x, centers, step):
    return jnp.exp(-(x[..., None] - centers) ** 2 / step ** 2)

def linear_centers(min_val=0.0, max_val=22.0, bins=64):
    centers = jnp.linspace(min_val, max_val, bins)
    step = abs(centers[1:] - centers[:-1])
    step = jnp.concatenate((step[:1], step), axis=0)
    return centers, step

def log_centers(min_val=0.0, max_val=200.0, bins=64, eps=1e-3):
    centers = jnp.exp(jnp.linspace(jnp.log(min_val + eps), jnp.log(max_val + eps), bins))
    step = abs(centers[1:] - centers[:-1])
    step = jnp.concatenate((step[:1], step), axis=0)
    return centers, step

def linlog_centers(min_val=0.0, mid_val=20.0, max_val=400.0, bins=64, eps=1e-3):
    centers = jnp.concatenate((
        jnp.linspace(min_val, mid_val, bins // 2),
        jnp.exp(jnp.linspace(jnp.log(mid_val + eps), jnp.log(max_val + eps), bins // 2 + 1)[1:])
    ), axis=0)
    step = abs(centers[1:] - centers[:-1])
    step = jnp.concatenate((step[:1], step), axis=0)
    return centers, step

def distance_rbf(distance, min_distance=0.0, max_distance=22.0, bins=64, centers=None):
    """Computes Gaussian RBF features of continuous inputs.
    
    Args:
        distance: array of distances to embed.
        min_distance: minimum distance to place RBFs.
        max_distance: maximum distance to place RBFs.
        bins: number of radial basis functions.

    Returns:
        Gaussian RBF embedding of `distance` with `bins` centers.
    """
    if centers is None:
        step = (max_distance - min_distance) / bins
        centers = min_distance + jnp.arange(bins) * step + step / 2
    else:
        # compute per-bin step, if centers are provided
        step = abs(centers[1:] - centers[:-1])
        step = jnp.concatenate((step[:1], step), axis=0)
    return gaussian_rbf(distance, centers, step)

def distance_one_hot(distance, min_distance=0.0, max_distance=22.0, bins=64):
    """Computes one-hot encoding of continuous inputs.

    Args:
        distance: array of distances to embed.
        min_distance: minimum distance for input binning.
        max_distance: maximum distance for input binning.
        bins: number of bins.

    Returns:
        One-hot encoding of `distance` with `bins` bins.
    """
    step = (max_distance - min_distance) / bins
    centers = min_distance + jnp.arange(bins) * step + step / 2
    argmin = jnp.argmin(abs(distance[..., None] - centers), axis=-1)
    return jax.nn.one_hot(argmin, bins, axis=-1)

def hl_gaussian(data, minimum=0.0, maximum=22.0, bins=64, sigma_ratio=1.0):
    """Computes HL-Gauss multihot encoding of continuous inputs.
    
    HL-Gauss embedding proposed by Farebrother et al. 2024 (arxiv.org/abs/2403.03950v1).
    Convolves a Gaussian with the input and bins the resulting probability distribution.
    
    Args:
        distance: array of distances to embed.
        min_distance: minimum distance for input binning.
        max_distance: maximum distance for input binning.
        bins: number of bins.
        sigma_ratio: scaling factor broadening or sharpening
            the gaussian convolved with the data.

    Returns:
        HL-Gauss encoding of `distance` with `bins` bins.
    """
    step = (maximum - minimum) / bins
    sigma = step * sigma_ratio
    def erf_aux(x, mu):
        return jax.scipy.special.erf((x - mu) / (jnp.sqrt(2) * sigma))
    def erfinv_aux(x, mu):
        return jax.scipy.special.erfinv(x) * (jnp.sqrt(2) * sigma) + mu
    # set an upper and lower bound for the input data
    # to stop the output from becoming NaN
    lower_bound = erfinv_aux(-0.999, minimum)
    upper_bound = erfinv_aux(0.999, maximum)
    data = jnp.clip(data, lower_bound, upper_bound)
    lower = jnp.arange(bins) * step
    upper = lower + step
    value = erf_aux(upper, data[..., None]) - erf_aux(lower, data[..., None])
    value /= erf_aux(maximum, data[..., None]) - erf_aux(minimum, data[..., None])
    return value

def spherical_harmonics(x, l_max=10):
    normed = x / jnp.maximum(jnp.linalg.norm(x, axis=-1)[..., None], 1e-6)
    colatitude_ = jnp.pi / 2 + jnp.arctan2(normed[..., 2], jnp.linalg.norm(normed[..., :2], axis=-1))
    longitude_ = jnp.pi + jnp.arctan2(normed[..., 1], normed[..., 0])
    upper_index = jnp.triu_indices(l_max + 1)
    colat = jax.scipy.special.lpmn_values(l_max, l_max, jnp.cos(colatitude_).reshape(-1), is_normalized=True)
    m = jnp.repeat(jnp.arange(l_max + 1)[:, None], l_max + 1, axis=1)
    m = jnp.triu(m)
    cos_val = colat * jnp.cos(m[:, :, None] * longitude_.reshape(-1)[None, None, :])
    sin_val = colat * jnp.sin(m[:, :, None] * longitude_.reshape(-1)[None, None, :])
    val = cos_val
    val = val[upper_index]
    val = jnp.moveaxis(val, 0, -1)
    val = val.reshape(*x.shape[:-1], -1)
    return val

def compute_pseudo_cb(positions):
    """Compute idealized CB atom positions.
    
    Args:
        positions: array of atom positions in atom14 order
            containing at least N, CA and C of shape (N, 3+, 3).

    Returns:
        Array of idealized CB atom positions of shape (N, 3).
    """
    n, ca, c = jnp.moveaxis(positions[..., :3, :], -2, 0)
    b = ca - n
    c = c - ca
    a = jnp.cross(b, c)
    const = [-0.58273431, 0.56802827, -0.54067466]
    return const[0] * a + const[1] * b + const[2] * c + ca

def axis_index(data: jnp.ndarray, axis=0):
    """Index along an axis of `data`.
    
    Args:
        data: input data array of shape (..., N, ...).
        axis: axis along which to construct an index.
    Returns:
        Index array containing values (0, ..., N-1).
    """
    return jnp.arange(data.shape[axis], dtype=jnp.int32)

def index_sum(data: jnp.ndarray,
              index: jnp.ndarray,
              mask: jnp.ndarray,
              apply_mask: bool = True) -> jnp.ndarray:
    """Sum array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Sum of array entries with the same index value, broadcasted
        to all entries with that index value.
        E.g. for values [1, 2, 3, 4, 5] and index [0, 0, 0, 1, 1]
        the result would be [6, 6, 6, 9, 9].
    """
    data = jnp.where(mask, data, 0)
    result = jnp.zeros_like(data).at[index].add(data)
    if not apply_mask:
        return result[index]
    return jnp.where(mask, result[index], 0)

def index_max(data: jnp.ndarray,
              index: jnp.ndarray,
              mask: jnp.ndarray,
              apply_mask: bool = True) -> jnp.ndarray:
    """Maximum of array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Maximum of array entries with the same index value, broadcasted
        to all entries with that index value.
        E.g. for values [1, 2, 3, 4, 5] and index [0, 0, 0, 1, 1]
        the result would be [3, 3, 3, 5, 5].
    """
    dmin = data.min()
    data = jnp.where(mask, data, dmin)
    result = jnp.full_like(data, dmin).at[index].max(data)
    if not apply_mask:
        return result[index]
    return jnp.where(mask, result[index], dmin)

def index_mean(data: jnp.ndarray,
               index: jnp.ndarray,
               mask: jnp.ndarray,
               weight: Optional[jnp.ndarray] = None,
               apply_mask: bool = True):
    """Mean of array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Mean of array entries with the same index value, broadcasted
        to all entries with that index value.
        E.g. for values [1, 2, 3, 4, 5] and index [0, 0, 0, 1, 1]
        the result would be [2, 2, 2, 4.5, 4.5].
    """
    if weight is not None:
        data *= weight
    data = jnp.where(mask, data, 0)
    result = jnp.zeros_like(data).at[index].add(data)
    position_weight = mask
    if weight is not None:
        position_weight = jnp.where(mask, weight, 0)
    result /= jnp.maximum(jnp.zeros_like(data).at[index].add(position_weight), 1e-6)
    if not apply_mask:
        return result[index]
    return jnp.where(mask, result[index], 0)

def index_var(data: jnp.ndarray,
              index: jnp.ndarray,
              mask: jnp.ndarray,
              apply_mask: bool = True):
    """Variance of array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Variance of array entries with the same index value, broadcasted
        to all entries with that index value.
    """
    ex2 = index_mean(data ** 2, index, mask, apply_mask=apply_mask)
    e2x = index_mean(data, index, mask, apply_mask=apply_mask) ** 2
    return ex2 - e2x

def index_std(data: jnp.ndarray,
              index: jnp.ndarray,
              mask: jnp.ndarray,
              apply_mask: bool = True,
              eps: Optional[float] = 1e-6):
    """Standard deviation of array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Standard deviation of array entries with the same index value, broadcasted
        to all entries with that index value.
    """
    return jnp.sqrt(index_var(data, index, mask, apply_mask) + eps)

def index_count(index, mask, apply_mask=True):
    """Count the number of entries with the same index value.
    
    Args:
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Count of index entries with the same value, broadcasted
        to all entries with that index value.
        E.g. for index [0, 0, 0, 1, 1] the result would be [3, 3, 3, 2, 2]
    """
    result = jnp.zeros_like(index).at[index].add(mask.astype(index.dtype))
    if not apply_mask:
        return result[index]
    return jnp.where(mask, result[index], 0)

def index_kabsch(x, y, index, mask, weight=None):
    """Compute rigid alignment parameters for two structures x and y.
    
    Args:
        x, y: atom positions of shape (N, 3).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        weight: optional array of importance weights for biasing alignment. Default: None.
    Returns:
        Alignment parameters (rotation, center of x, center of y) for atom position
        arrays x and y. Parts of x and y with different index are aligned separately.
    """
    if weight is not None:
        weight = weight[:, None]
    x_center = index_mean(x, index, mask[:, None], weight=weight)
    y_center = index_mean(y, index, mask[:, None], weight=weight)
    x -= x_center
    y -= y_center
    if weight is None:
        weight = jnp.ones_like(index, dtype=jnp.float32)[:, None]
    covariance = index_sum(
        weight[:, :, None] * x[:, :, None] * y[:, None, :],
        index, mask[:, None, None])
    u, _, v = jnp.linalg.svd(
        jax.lax.stop_gradient(covariance), compute_uv=True, full_matrices=True)
    det_sign = jnp.linalg.det(u) * jnp.linalg.det(v)
    det_flip = jnp.ones((v.shape[0], 3)).at[:, -1].set(det_sign)
    rot = jnp.einsum("...ak,...kb->...ba", u * det_flip[:, None, :], v)
    return rot, x_center, y_center

def index_align(x, y, index, mask, weight=None):
    """Rigid align two structures x and y.
    
    Args:
        x, y: atom positions of shape (N, ..., 3).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        weight: optional array of importance weights for biasing alignment. Default: None.
    Returns:
        x aligned to y.
    """
    return_vec3 = False
    if isinstance(x, Vec3Array):
        x = x.to_array()
        return_vec3 = True
    if isinstance(y, Vec3Array):
        y = y.to_array()
        return_vec3 = True
    rot, x_center, y_center = index_kabsch(
        x[:, 1], y[:, 1], index, mask, weight=weight)
    result = jnp.einsum(
        "...ak,...ik->...ia", rot, (x - x_center[:, None])) + y_center[:, None]
    if return_vec3:
        result = Vec3Array.from_array(result)
    return result

def apply_alignment(x, kabsch_data):
    """Apply alignment parameters to a structure.
    
    Args:
        x: atom positions of shape (N, ..., 3).
        kabsch_data: output of index_kabsch.
    Returns:
        x transformed according to kabsch_data.
    """
    rot, x_center, y_center = kabsch_data
    # delta = jnp.einsum(
    #      "...ak,...k->...a", jnp.swapaxes(rot, -1, -2), y_center) - x_center
    return_vec3 = False
    if isinstance(x, Vec3Array):
        x = x.to_array()
        return_vec3 = True
    result = jnp.einsum(
        "...ak,...ik->...ia", rot, (x - x_center[:, None])) + y_center[:, None]
    # result = jnp.einsum(
    #     "...ak,...ik->...ia", rot, x + delta[:, None])
    if return_vec3:
        result = Vec3Array.from_array(result)
    return result

def unique_chain(chain, batch):
    """Compute a unique chain index given a batch of chains.
    
    Args:
        chain: chain index.
        batch: batch index.
    Returns:
        Chain index where no two chains share their index.
    """
    def unique_chain_body(carry, data):
        chain, batch = data
        prev_chain, prev_batch, current_index = carry
        new_chain = (chain != prev_chain) + (batch != prev_batch) > 0
        new_carry = jnp.where(
            new_chain[None],
            jnp.array([chain, batch, current_index + 1]),
            carry)
        return new_carry, new_carry[-1]
    _, chain = jax.lax.scan(
        unique_chain_body,
        -jnp.ones((3,), dtype=chain.dtype),
        (chain, batch))
    return chain

def positions_to_ncacocb(pos: jnp.ndarray):
    """Compute N, CA, C, O, CB positions for atom14 positions.
    
    Args:
        pos: atom positions in atom14 format containing at least N, CA, C and O.
    Returns:
        Atom positions for N, CA, C, O and idealised CB.
    """
    cb = compute_pseudo_cb(pos)
    return jnp.concatenate((pos[:, :4], cb[..., None, :]), axis=-2)

def replace_masked_with(pos: jnp.ndarray, # (..., N, 3)
                        atom_mask: jnp.ndarray, # (.... N)
                        replacement: jnp.ndarray # (..., 3)
                       ) -> jnp.ndarray: # (..., N, 3)
    """Replace masked atom positions with replacement positions."""
    return jnp.where(atom_mask[..., None], pos, replacement)

def assign_sse(pos, batch, mask):
    """Assign secondary structure using P-SEA.
    
    Implements the secondary structure assignment from Labesse et al. 1997
    (pubmed.ncbi.nlm.nih.gov/9183534/)

    Args:
        pos: atom positions in atom14 format, containing at least CA atoms.
        batch: batch index.
        mask: residue mask.
    Returns:
        3-state secondary structure assignment (0: loop, 1: helix, 2: strand);
        secondary structure blocks; block adjacency matrix.
    """
    pos = pos[:, 1]
    d2 = jnp.concatenate((jnp.zeros((1,)), jnp.linalg.norm(pos[2:] - pos[:-2], axis=-1), jnp.zeros((1,))), axis=0)
    d3 = jnp.concatenate((jnp.zeros((1,)), jnp.linalg.norm(pos[3:] - pos[:-3], axis=-1), jnp.zeros((2,))), axis=0)
    d4 = jnp.concatenate((jnp.zeros((1,)), jnp.linalg.norm(pos[4:] - pos[:-4], axis=-1), jnp.zeros((3,))), axis=0)
    tau = jnp.concatenate((jnp.zeros((1,)), bond_angle(pos[:-2], pos[1:-1], pos[2:]), jnp.zeros((1,))), axis=0)
    alpha = jnp.concatenate((jnp.zeros((1,)), dihedral_angle(pos[:-3], pos[1:-2], pos[2:-1], pos[3:]), jnp.zeros((2,))), axis=0)
    helix_tau = (89 - 12 <= tau) * (tau <= 89 + 12)
    helix_alpha = (50 - 20 <= alpha) * (alpha <= 50 + 20)
    helix_d2 = (5.5 - 0.5 <= d2) * (d2 <= 5.5 + 0.5)
    helix_d3 = (5.3 - 0.5 <= d3) * (d3 <= 5.3 + 0.5)
    helix_d4 = (6.4 - 0.6 <= d4) * (d4 <= 6.4 + 0.6)
    sheet_tau = (124 - 14 <= tau) * (tau <= 124 + 14)
    sheet_alpha = (-170 - 45 <= alpha) * (alpha <= -170 + 45)
    sheet_d2 = (6.7 - 0.6 <= d2) * (d2 <= 6.7 + 0.6)
    sheet_d3 = (9.9 - 0.9 <= d3) * (d3 <= 9.9 + 0.9)
    sheet_d4 = (12.4 - 1.1 <= d4) * (d4 <= 12.4 + 1.1)
    helix_init = (helix_tau * helix_alpha + helix_d3 * helix_d4) > 0
    helix_extend = (helix_init + helix_tau + helix_d3) > 0
    sheet_init = (sheet_tau * sheet_alpha + sheet_d2 * sheet_d3 * sheet_d4) > 0
    sheet_extend = (sheet_init + sheet_d3) > 0
    def body(carry, data):
        result = 0
        result = jnp.where(data["h_init"], 1, result)
        result = jnp.where(data["h_extend"] * (carry == 1), 1, result)
        result = jnp.where(data["e_init"], 2, result)
        result = jnp.where(data["e_extend"] * (carry == 2), 2, result)
        return result, result
    _, index = jax.lax.scan(body, jnp.zeros((), dtype=jnp.int32),
                             dict(h_init=helix_init, h_extend=helix_extend,
                                  e_init=sheet_init, e_extend=sheet_extend))
    def block_iter(carry, data):
        state, block_index = carry
        new_state = data
        block_index = jnp.where(new_state != state, block_index + 1, block_index)
        return (new_state, block_index), block_index
    _, blocks = jax.lax.scan(block_iter, (index[0], jnp.zeros((), dtype=jnp.int32)), index)
    loop = index == 0
    pair_mask = (batch[:, None] == batch[None, :]) * mask[:, None] * mask[None, :]
    same_block = blocks[:, None] == blocks[None, :]
    sheet_sheet = (index[:, None] == 2) * (index[None, :] == 2)
    distance = jnp.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    block_distance = (1e6 * jnp.ones_like(distance)).at[blocks[:, None], blocks[None, :]].min(distance)[blocks[:, None], blocks[None, :]]
    block_adjacency = jnp.where(sheet_sheet, block_distance <= 8, block_distance <= 8)
    block_adjacency = jnp.where(same_block + loop[:, None] + loop[None, :] > 0, 0, block_adjacency)
    block_adjacency = jnp.where(pair_mask, block_adjacency, 0)
    secondary_structure = index
    return secondary_structure, blocks, block_adjacency

POLAR_THRESHOLD = 3.0
CONTACT_THRESHOLD = 6.0
def interactions(coords, sse, mask, resi, chain, batch, block_noise=None):
    """Compute interactions between protein chains.
    
    Args:
        coords: atom coordinates in atom14 format.
        sse: secondary structure assignment.
        mask: residue mask.
        resi, chain, batch: residue, chain and batch index.
    Returns:
        Dictionary of interaction hotspots, chain contacts,
        and block contacts.
    """
    same_batch = batch[:, None] == batch[None, :]
    same_chain = (chain[:, None] == chain[None, :]) * same_batch
    pair_mask = mask[:, None, :, None] * mask[None, :, None, :]
    pair_mask *= same_batch[:, :, None, None]
    distances = jnp.linalg.norm(coords[:, None, :, None] - coords[None, :, None, :], axis=-1)
    distances = jnp.where(pair_mask, distances, jnp.inf)
    contact_interaction = distances.min(axis=(-1, -2)) <= CONTACT_THRESHOLD

    size = resi.shape[0]
    # which chains are in contact with each other at all?
    chain_contact = jnp.zeros((size // 10, size // 10), dtype=jnp.bool_).at[chain[:, None], chain[None, :]].max(contact_interaction)
    chain_contact = jnp.where(jnp.eye(size // 10, size // 10), 0, chain_contact)
    chain_contact = chain_contact[chain[:, None], chain[None, :]]
    chain_contact = jnp.where(same_batch, chain_contact, 0)
    chain_contact = jnp.where(same_chain, 0, chain_contact)
    # which residues on this chain interact with anything else?
    hotspot = contact_interaction.any(axis=1)
    # which residues on another chain are interaction hotspots as seen by this chain?
    # take the union of all contacts for each chain across all positions in all other chains.
    relative_hotspot = jnp.zeros((size // 10, contact_interaction.shape[1]), dtype=jnp.bool_).at[chain, :].max(contact_interaction)[chain]
    # the transpose of this tells us which hotspot residues on this chain interact with another chain
    # we should include both in attention computations, as any kind of hotspot specification should
    # be seen by both chains in the granularity it is provided.
    # E.g. if we specify 5 hotspots on chain A that should be bound by any residue on chain B,
    # then chain A should know that these 5 hotspots are going to be bound by at least one residue on chain B
    # and all residues on chain B should know that at least one of them should bind anywhere on chain A

    def block_iter(carry, data):
        state, block_index = carry
        new_state, flip = data
        block_index = jnp.where((new_state != state) + flip, block_index + 1, block_index)
        return (new_state, block_index), block_index
    def blockify(pairwise, blocks):
        same_block = blocks[:, None] == blocks[None, :]
        result = jnp.zeros_like(pairwise).at[blocks[:, None], blocks[None, :]].max(pairwise)[blocks[:, None], blocks[None, :]]
        return jnp.where(same_block, 0, result)
    flip = block_noise if block_noise is not None else jnp.zeros_like(sse, dtype=jnp.bool_)
    _, blocks = jax.lax.scan(block_iter, (sse[0], jnp.zeros((), dtype=jnp.int32)), (sse, flip))
    block_contact = blockify(contact_interaction, blocks)
    return dict(
        hotspot=hotspot,
        chain_contact=chain_contact,
        relative_hotspot=relative_hotspot,
        block_contact=block_contact
    )

def unit_sphere(n):
    """Generates n points on the surface of the unit sphere."""
    dl = np.pi * (3 - 5 ** 0.5)
    dz = 2.0 / n

    indices = np.arange(n)
    z = 1 - dz / 2 - indices * dz
    longitude = indices * dl
    r = (1 - z ** 2) ** 0.5
    coords = np.stack((
        np.cos(longitude) * r,
        np.sin(longitude) * r,
        z), axis=-1)
    return coords

ATOM14_RADIUS=np.array([
    [
        residue_constants.van_der_waals_radius[c[0]] + 1.4
        if c else 0.0
        for c in residue_constants.restype_name_to_atom14_names[res]
    ]
    for res in residue_constants.restype_name_to_atom14_names
])
def fast_sasa(pos, atom_mask, aatype, batch, atom_radius=ATOM14_RADIUS, n=20, neighbours=20):
    """Quick and dirty SASA estimate.
    
    Args:
        pos: atom positions in atom14 format.
        atom_mask: atom mask.
        aatype: amino acid identity for each residue.
        batch: batch_index.
        atom_radius: dictionary of atom radii. Default: ATOM14_RADIUS.
        n: number of points on each unit sphere.
        neighbours: number of neighbour amino acids for SASA computation.
    Returns:
        Per residue SASA estimate.
    """
    mask = atom_mask.any(axis=1)
    valid = (batch[:, None] == batch[None, :]) * (mask[:, None] * mask[None, :])
    radius = atom_radius[aatype] * atom_mask
    spheres = pos[:, :, None, :] + radius[:, :, None, None] * unit_sphere(n)[None, None, :, :]
    aa_dist = jnp.linalg.norm(pos[:, None, 1] - pos[None, :, 1], axis=-1)
    aa_dist = jnp.where(valid, aa_dist, jnp.inf)
    neighbours = jnp.argsort(aa_dist, axis=1)[:, :neighbours]
    index = axis_index(neighbours, 0)
    neighbours = jnp.where(valid[index[:, None], neighbours], neighbours, -1)
    distances = jnp.linalg.norm(spheres[:, None, :, None, :, :] - pos[neighbours, None, :, None, :], axis=-1)
    drop = (distances < radius[neighbours, None, :, None])
    index = jnp.arange(atom_radius.shape[0])
    drop = drop.at[:, 0, index, index].set(0)
    drop = drop.any(axis=(1, 3))
    count = n * atom_mask - (atom_mask[..., None] * drop).sum(axis=2)
    bare_radius = 4 * jnp.pi * radius ** 2
    surf = bare_radius * count / n
    surf = surf.sum(axis=1) / atom_mask.sum(axis=1)
    return surf
