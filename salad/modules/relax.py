# TODO: atomistic relaxation function

from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk

# alphafold dependencies
from alphafold.model.geometry import Vec3Array
from alphafold.model.all_atom_multimer import get_atom14_mask

# basic module imports
from salad.modules.basic import (
    Linear, MLP, init_glorot, init_relu,
    init_zeros, init_linear, block_stack
)
from salad.modules.transformer import (
    resi_dual, prenorm_skip, resi_dual_input, prenorm_input)

# import geometry utils
from salad.modules.utils.geometry import (
    index_mean, index_sum, index_count, extract_aa_frames,
    extract_neighbours, distance_rbf, hl_gaussian, distance_one_hot,
    unique_chain, positions_to_ncacocb, axis_index,
    single_protein_sidechains, compute_pseudo_cb,
    get_random_neighbours, get_spatial_neighbours,
    get_index_neighbours, get_neighbours, index_align)

from salad.modules.utils.dssp import assign_dssp, drop_dssp

# TODO add pre Encoder and post Encoder augmentation to semi-equivariant mode
from salad.modules.structure_autoencoder import (
    structure_augmentation, structure_augmentation_params,
    apply_structure_augmentation, apply_inverse_structure_augmentation,
    semiequivariant_update_positions,
    InnerDistogram, extract_dmap_neighbours)

# import violation loss
from salad.modules.utils.alphafold_loss import violation_loss

# sparse geometric module imports
from salad.modules.geometric import (
    SparseStructureAttention,
    SemiEquivariantSparseStructureAttention,
    VectorLinear,
    vector_mean_norm,
    sequence_relative_position, distance_features,
    direction_features, pair_vector_features,
    position_rotation_features
)

# diffusion processes
from salad.modules.utils.diffusion import (
    diffuse_coordinates_edm, diffuse_atom_cloud,
    diffuse_sequence, fourier_time_embedding
)

class RelaxBlock(hk.Module):
    def __init__(self, config, name: str | None = None):
        super().__init__(name)
        self.config = config
