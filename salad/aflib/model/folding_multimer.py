# adapted from the AlphaFold 2 codebase (https://github.com/google-deepmind/alphafold)
#
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules and utilities for the structure module in the multimer system."""

import functools
import numbers
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from salad.aflib.common import residue_constants
from salad.aflib.model import all_atom_multimer
from salad.aflib.model import geometry
from salad.aflib.model import utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np


EPSILON = 1e-8
Float = Union[float, jnp.ndarray]


def squared_difference(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes Squared difference between two arrays."""
  return jnp.square(x - y)


def make_backbone_affine(
    positions: geometry.Vec3Array,
    mask: jnp.ndarray,
    aatype: jnp.ndarray,
    ) -> Tuple[geometry.Rigid3Array, jnp.ndarray]:
  """Make backbone Rigid3Array and mask."""
  del aatype
  a = residue_constants.atom_order['N']
  b = residue_constants.atom_order['CA']
  c = residue_constants.atom_order['C']

  rigid_mask = (mask[:, a] * mask[:, b] * mask[:, c]).astype(
      jnp.float32)

  rigid = all_atom_multimer.make_transform_from_reference(
      a_xyz=positions[:, a], b_xyz=positions[:, b], c_xyz=positions[:, c])

  return rigid, rigid_mask

def compute_atom14_gt(
    aatype: jnp.ndarray,
    all_atom_positions: geometry.Vec3Array,
    all_atom_mask: jnp.ndarray,
    pred_pos: geometry.Vec3Array
) -> Tuple[geometry.Vec3Array, jnp.ndarray, jnp.ndarray]:
  """Find atom14 positions, this includes finding the correct renaming."""
  gt_positions, gt_mask = all_atom_multimer.atom37_to_atom14(
      aatype, all_atom_positions,
      all_atom_mask)
  alt_gt_positions, alt_gt_mask = all_atom_multimer.get_alt_atom14(
      aatype, gt_positions, gt_mask)
  atom_is_ambiguous = all_atom_multimer.get_atom14_is_ambiguous(aatype)

  alt_naming_is_better = all_atom_multimer.find_optimal_renaming(
      gt_positions=gt_positions,
      alt_gt_positions=alt_gt_positions,
      atom_is_ambiguous=atom_is_ambiguous,
      gt_exists=gt_mask,
      pred_positions=pred_pos)

  use_alt = alt_naming_is_better[:, None]

  gt_mask = (1. - use_alt) * gt_mask + use_alt * alt_gt_mask
  gt_positions = (1. - use_alt) * gt_positions + use_alt * alt_gt_positions

  return gt_positions, gt_mask, alt_naming_is_better


def backbone_loss(gt_rigid: geometry.Rigid3Array,
                  gt_frames_mask: jnp.ndarray,
                  gt_positions_mask: jnp.ndarray,
                  target_rigid: geometry.Rigid3Array,
                  config: ml_collections.ConfigDict,
                  pair_mask: jnp.ndarray
                  ) -> Tuple[Float, jnp.ndarray]:
  """Backbone FAPE Loss."""
  loss_fn = functools.partial(
      all_atom_multimer.frame_aligned_point_error,
      l1_clamp_distance=config.atom_clamp_distance,
      length_scale=config.loss_unit_distance)

  loss_fn = jax.vmap(loss_fn, (0, None, None, 0, None, None, None))
  fape = loss_fn(target_rigid, gt_rigid, gt_frames_mask,
                 target_rigid.translation, gt_rigid.translation,
                 gt_positions_mask, pair_mask)

  return jnp.mean(fape), fape[-1]


def compute_frames(
    aatype: jnp.ndarray,
    all_atom_positions: geometry.Vec3Array,
    all_atom_mask: jnp.ndarray,
    use_alt: jnp.ndarray
    ) -> Tuple[geometry.Rigid3Array, jnp.ndarray]:
  """Compute Frames from all atom positions.

  Args:
    aatype: array of aatypes, int of [N]
    all_atom_positions: Vector of all atom positions, shape [N, 37]
    all_atom_mask: mask, shape [N]
    use_alt: whether to use alternative orientation for ambiguous aatypes
             shape [N]
  Returns:
    Rigid corresponding to Frames w shape [N, 8],
    mask which Rigids are present w shape [N, 8]
  """
  frames_batch = all_atom_multimer.atom37_to_frames(aatype, all_atom_positions,
                                                    all_atom_mask)
  gt_frames = frames_batch['rigidgroups_gt_frames']
  alt_gt_frames = frames_batch['rigidgroups_alt_gt_frames']
  use_alt = use_alt[:, None]

  renamed_gt_frames = jax.tree.map(
      lambda x, y: (1. - use_alt) * x + use_alt * y, gt_frames, alt_gt_frames)

  return renamed_gt_frames, frames_batch['rigidgroups_gt_exists']


def sidechain_loss(gt_frames: geometry.Rigid3Array,
                   gt_frames_mask: jnp.ndarray,
                   gt_positions: geometry.Vec3Array,
                   gt_mask: jnp.ndarray,
                   pred_frames: geometry.Rigid3Array,
                   pred_positions: geometry.Vec3Array,
                   config: ml_collections.ConfigDict
                   ) -> Dict[str, jnp.ndarray]:
  """Sidechain Loss using cleaned up rigids."""

  flat_gt_frames = jax.tree.map(jnp.ravel, gt_frames)
  flat_frames_mask = jnp.ravel(gt_frames_mask)

  flat_gt_positions = jax.tree.map(jnp.ravel, gt_positions)
  flat_positions_mask = jnp.ravel(gt_mask)

  # Compute frame_aligned_point_error score for the final layer.
  def _slice_last_layer_and_flatten(x):
    return jnp.ravel(x[-1])

  flat_pred_frames = jax.tree.map(_slice_last_layer_and_flatten, pred_frames)
  flat_pred_positions = jax.tree.map(_slice_last_layer_and_flatten,
                                     pred_positions)
  fape = all_atom_multimer.frame_aligned_point_error(
      pred_frames=flat_pred_frames,
      target_frames=flat_gt_frames,
      frames_mask=flat_frames_mask,
      pred_positions=flat_pred_positions,
      target_positions=flat_gt_positions,
      positions_mask=flat_positions_mask,
      pair_mask=None,
      length_scale=config.sidechain.loss_unit_distance,
      l1_clamp_distance=config.sidechain.atom_clamp_distance)

  return {
      'fape': fape,
      'loss': fape}


def structural_violation_loss(mask: jnp.ndarray,
                              violations: Mapping[str, Float],
                              config: ml_collections.ConfigDict
                              ) -> Float:
  """Computes Loss for structural Violations."""
  # Put all violation losses together to one large loss.
  num_atoms = jnp.sum(mask).astype(jnp.float32) + 1e-6
  between_residues = violations['between_residues']
  within_residues = violations['within_residues']
  return (config.structural_violation_loss_weight *
          (between_residues['bonds_c_n_loss_mean'] +
           between_residues['angles_ca_c_n_loss_mean']  +
           between_residues['angles_c_n_ca_loss_mean'] +
           jnp.sum(between_residues['clashes_per_atom_loss_sum'] +
                   within_residues['per_atom_loss_sum']) / num_atoms
           ))


def find_structural_violations(
    aatype: jnp.ndarray,
    residue_index: jnp.ndarray,
    mask: jnp.ndarray,
    pred_positions: geometry.Vec3Array,  # (N, 14)
    config: ml_collections.ConfigDict,
    asym_id: jnp.ndarray,
    ) -> Dict[str, Any]:
  """Computes several checks for structural Violations."""

  # Compute between residue backbone violations of bonds and angles.
  connection_violations = all_atom_multimer.between_residue_bond_loss(
      pred_atom_positions=pred_positions,
      pred_atom_mask=mask.astype(jnp.float32),
      residue_index=residue_index.astype(jnp.float32),
      aatype=aatype,
      tolerance_factor_soft=config.violation_tolerance_factor,
      tolerance_factor_hard=config.violation_tolerance_factor)

  # Compute the van der Waals radius for every atom
  # (the first letter of the atom name is the element type).
  # shape (N, 14)
  atomtype_radius = jnp.array([
      residue_constants.van_der_waals_radius[name[0]]
      for name in residue_constants.atom_types
  ])
  residx_atom14_to_atom37 = all_atom_multimer.get_atom14_to_atom37_map(aatype)
  atom_radius = mask * utils.batched_gather(atomtype_radius,
                                            residx_atom14_to_atom37)

  # Compute the between residue clash loss.
  between_residue_clashes = all_atom_multimer.between_residue_clash_loss(
      pred_positions=pred_positions,
      atom_exists=mask,
      atom_radius=atom_radius,
      residue_index=residue_index,
      overlap_tolerance_soft=config.clash_overlap_tolerance,
      overlap_tolerance_hard=config.clash_overlap_tolerance,
      asym_id=asym_id)

  # Compute all within-residue violations (clashes,
  # bond length and angle violations).
  restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
      overlap_tolerance=config.clash_overlap_tolerance,
      bond_length_tolerance_factor=config.violation_tolerance_factor)
  dists_lower_bound = utils.batched_gather(restype_atom14_bounds['lower_bound'],
                                           aatype)
  dists_upper_bound = utils.batched_gather(restype_atom14_bounds['upper_bound'],
                                           aatype)
  within_residue_violations = all_atom_multimer.within_residue_violations(
      pred_positions=pred_positions,
      atom_exists=mask,
      dists_lower_bound=dists_lower_bound,
      dists_upper_bound=dists_upper_bound,
      tighten_bounds_for_loss=0.0)

  # Combine them to a single per-residue violation mask (used later for LDDT).
  per_residue_violations_mask = jnp.max(jnp.stack([
      connection_violations['per_residue_violation_mask'],
      jnp.max(between_residue_clashes['per_atom_clash_mask'], axis=-1),
      jnp.max(within_residue_violations['per_atom_violations'],
              axis=-1)]), axis=0)

  return {
      'between_residues': {
          'bonds_c_n_loss_mean':
              connection_violations['c_n_loss_mean'],  # ()
          'angles_ca_c_n_loss_mean':
              connection_violations['ca_c_n_loss_mean'],  # ()
          'angles_c_n_ca_loss_mean':
              connection_violations['c_n_ca_loss_mean'],  # ()
          'connections_per_residue_loss_sum':
              connection_violations['per_residue_loss_sum'],  # (N)
          'connections_per_residue_violation_mask':
              connection_violations['per_residue_violation_mask'],  # (N)
          'clashes_mean_loss':
              between_residue_clashes['mean_loss'],  # ()
          'clashes_per_atom_loss_sum':
              between_residue_clashes['per_atom_loss_sum'],  # (N, 14)
          'clashes_per_atom_clash_mask':
              between_residue_clashes['per_atom_clash_mask'],  # (N, 14)
      },
      'within_residues': {
          'per_atom_loss_sum':
              within_residue_violations['per_atom_loss_sum'],  # (N, 14)
          'per_atom_violations':
              within_residue_violations['per_atom_violations'],  # (N, 14),
      },
      'total_per_residue_violations_mask':
          per_residue_violations_mask,  # (N)
  }


def compute_violation_metrics(
    residue_index: jnp.ndarray,
    mask: jnp.ndarray,
    seq_mask: jnp.ndarray,
    pred_positions: geometry.Vec3Array,  # (N, 14)
    violations: Mapping[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
  """Compute several metrics to assess the structural violations."""
  ret = {}
  between_residues = violations['between_residues']
  within_residues = violations['within_residues']
  extreme_ca_ca_violations = all_atom_multimer.extreme_ca_ca_distance_violations(
      positions=pred_positions,
      mask=mask.astype(jnp.float32),
      residue_index=residue_index.astype(jnp.float32))
  ret['violations_extreme_ca_ca_distance'] = extreme_ca_ca_violations
  ret['violations_between_residue_bond'] = utils.mask_mean(
      mask=seq_mask,
      value=between_residues['connections_per_residue_violation_mask'])
  ret['violations_between_residue_clash'] = utils.mask_mean(
      mask=seq_mask,
      value=jnp.max(between_residues['clashes_per_atom_clash_mask'], axis=-1))
  ret['violations_within_residue'] = utils.mask_mean(
      mask=seq_mask,
      value=jnp.max(within_residues['per_atom_violations'], axis=-1))
  ret['violations_per_residue'] = utils.mask_mean(
      mask=seq_mask, value=violations['total_per_residue_violations_mask'])
  return ret


def supervised_chi_loss(
    sequence_mask: jnp.ndarray,
    target_chi_mask: jnp.ndarray,
    aatype: jnp.ndarray,
    target_chi_angles: jnp.ndarray,
    pred_angles: jnp.ndarray,
    unnormed_angles: jnp.ndarray,
    config: ml_collections.ConfigDict) -> Tuple[Float, Float, Float]:
  """Computes loss for direct chi angle supervision."""
  eps = 1e-6
  chi_mask = target_chi_mask.astype(jnp.float32)

  pred_angles = pred_angles[:, :, 3:]

  residue_type_one_hot = jax.nn.one_hot(
      aatype, residue_constants.restype_num + 1, dtype=jnp.float32)[None]
  chi_pi_periodic = jnp.einsum('ijk, kl->ijl', residue_type_one_hot,
                               jnp.asarray(residue_constants.chi_pi_periodic))

  true_chi = target_chi_angles[None]
  sin_true_chi = jnp.sin(true_chi)
  cos_true_chi = jnp.cos(true_chi)
  sin_cos_true_chi = jnp.stack([sin_true_chi, cos_true_chi], axis=-1)

  # This is -1 if chi is pi periodic and +1 if it's 2 pi periodic
  shifted_mask = (1 - 2 * chi_pi_periodic)[..., None]
  sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi

  sq_chi_error = jnp.sum(
      squared_difference(sin_cos_true_chi, pred_angles), -1)
  sq_chi_error_shifted = jnp.sum(
      squared_difference(sin_cos_true_chi_shifted, pred_angles), -1)
  sq_chi_error = jnp.minimum(sq_chi_error, sq_chi_error_shifted)

  sq_chi_loss = utils.mask_mean(mask=chi_mask[None], value=sq_chi_error)
  angle_norm = jnp.sqrt(jnp.sum(jnp.square(unnormed_angles), axis=-1) + eps)
  norm_error = jnp.abs(angle_norm - 1.)
  angle_norm_loss = utils.mask_mean(mask=sequence_mask[None, :, None],
                                    value=norm_error)
  loss = (config.chi_weight * sq_chi_loss
          + config.angle_norm_weight * angle_norm_loss)
  return loss, sq_chi_loss, angle_norm_loss


def l2_normalize(x: jnp.ndarray,
                 axis: int = -1,
                 epsilon: float = 1e-12
                 ) -> jnp.ndarray:
  return x / jnp.sqrt(
      jnp.maximum(jnp.sum(x**2, axis=axis, keepdims=True), epsilon))


def get_renamed_chi_angles(aatype: jnp.ndarray,
                           chi_angles: jnp.ndarray,
                           alt_is_better: jnp.ndarray
                           ) -> jnp.ndarray:
  """Return renamed chi angles."""
  chi_angle_is_ambiguous = utils.batched_gather(
      jnp.array(residue_constants.chi_pi_periodic, dtype=jnp.float32), aatype)
  alt_chi_angles = chi_angles + np.pi * chi_angle_is_ambiguous
  # Map back to [-pi, pi].
  alt_chi_angles = alt_chi_angles - 2 * np.pi * (alt_chi_angles > np.pi).astype(
      jnp.float32)
  alt_is_better = alt_is_better[:, None]
  return (1. - alt_is_better) * chi_angles + alt_is_better * alt_chi_angles
