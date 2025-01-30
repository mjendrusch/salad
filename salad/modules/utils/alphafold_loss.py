# TODO TODO TODO: clean this shit up

# module packaging AlphaFold loss into managable
# functions / sublosses

from functools import partial
import numbers
from typing import Any, Dict, Mapping, Text, Tuple, Optional
from typing_extensions import TypedDict
from alphafold.model.geometry.rigid_matrix_vector import Float

import jax
import jax.numpy as jnp

import haiku as hk

from alphafold.model import utils as afutils
from alphafold.model import geometry
from alphafold.model import folding_multimer as fmm
from alphafold.model import all_atom_multimer
from alphafold.common import residue_constants
from alphafold.model.all_atom_multimer import (
  atom37_to_atom14, get_alt_atom14, get_atom14_is_ambiguous,
  find_optimal_renaming)

from salad.modules.utils.collections import dotdict
from salad.modules.utils.geometry import (
  Vec3Array, Rigid3Array, extract_aa_frames)

class BatchType(TypedDict):
  aatype: jnp.ndarray
  all_atom_positions: jnp.ndarray
  all_atom_mask: jnp.ndarray
  seq_mask: jnp.ndarray
  residue_index: jnp.ndarray

class SidechainType(TypedDict):
  angles_sin_cos: jnp.ndarray
  unnormalized_angles_sin_cos: jnp.ndarray
  atom_pos: geometry.Vec3Array
  frames: geometry.Rigid3Array

class ValueType(TypedDict):
  final_atom14_positions: jnp.ndarray
  sidechains: SidechainType
  traj: jnp.ndarray

def backbone_fape(prediction: Vec3Array, ground_truth: Vec3Array, batch, chain, mask):
    prediction = prediction.to_array().reshape(-1, *prediction.shape[2:], 3)
    prediction = Vec3Array.from_array(prediction)
    pframes, _ = extract_aa_frames(prediction)
    pframes = pframes.to_array()
    pframes = pframes.reshape(*prediction_shape[:-1], *pframes.shape[-2:])
    pframes = Rigid3Array.from_array(pframes)
    prediction = prediction.to_array().reshape(*prediction_shape, 3)
    prediction = Vec3Array.from_array(prediction)
    gframes, _ = extract_aa_frames(ground_truth)
    pglobal = pframes[:, :, None, None].apply_inverse_to_point(prediction[:, None, :, :])
    gglobal = gframes[:, None, None].apply_inverse_to_point(ground_truth[None, :, :])
    fape = jnp.where(pair_mask[None, ...], (pglobal - gglobal[None]).norm(), 0)
    fape = fape.sum(axis=-1) / jnp.maximum(pair_mask[None, ...].sum(axis=-1), 1e-6)
    rotamer_error = jnp.diagonal(fape, 0, -2, -1)
    rotamer_loss = rotamer_error.sum(axis=-1) / jnp.maximum(all_atom_mask.any(axis=-1).sum(axis=-1), 1e-6)
    rotamer_error = rotamer_error[-1]
    pair_mask = pair_mask.any(axis=-1)
    final_ae = fape[-1]
    same_chain = (chain[:, None] == chain[None, :]) * pair_mask
    other_chain = (chain[:, None] != chain[None, :]) * pair_mask
    intra_coin = jnp.zeros_like(jax.random.bernoulli(hk.next_rng_key(), 0.1, batch.shape))
    inter_coin = jnp.zeros_like(jax.random.bernoulli(hk.next_rng_key(), 0.1, batch.shape))
    intra_fape = jnp.where(same_chain[None], jnp.clip(fape, 0.0, jnp.where(intra_coin, jnp.inf, 10.0)), 0).sum(axis=(-1, -2))
    intra_fape /= jnp.maximum((pair_mask * same_chain)[None].sum(axis=(-1, -2)), 1e-6)
    intra_fape /= 10.0
    inter_fape = jnp.where(other_chain[None], jnp.clip(fape, 0.0, jnp.where(inter_coin, jnp.inf, 30.0)), 0).sum(axis=(-1, -2))
    inter_fape /= jnp.maximum((pair_mask * other_chain)[None].sum(axis=(-1, -2)), 1e-6)
    inter_fape /= 30.0
    fape = intra_fape + inter_fape
    return fape, rotamer_loss, final_ae, rotamer_error

def all_atom_fape(prediction: Vec3Array, ground_truth: Vec3Array, aa_type: jnp.ndarray,
                  batch: jnp.ndarray, chain: jnp.ndarray, all_atom_mask: jnp.ndarray,
                  intra_clip=10.0, inter_clip=30.0, p_clip=0.9, sidechain=True) -> jnp.ndarray:
    def get_renamings(aa_type, ground_truth, all_atom_mask, prediction):
        if ground_truth.shape[-1] != 14:
            gt_positions, gt_mask = atom37_to_atom14(aa_type, ground_truth, all_atom_mask)
        else:
            gt_positions = ground_truth
            gt_mask = all_atom_mask
        alt_gt_positions, alt_gt_mask = get_alt_atom14(aa_type, gt_positions, gt_mask)
        atom_is_ambiguous = get_atom14_is_ambiguous(aa_type)
        alt_naming_is_better = find_optimal_renaming(
            gt_positions=gt_positions,
            alt_gt_positions=alt_gt_positions,
            atom_is_ambiguous=atom_is_ambiguous,
            gt_exists=gt_mask,
            pred_positions=prediction)
        use_alt = alt_naming_is_better[:, None]

        all_atom_mask = (1. - use_alt) * gt_mask + use_alt * alt_gt_mask
        gt_positions = (1. - use_alt) * gt_positions + use_alt * alt_gt_positions
        return gt_positions, all_atom_mask
    if sidechain:
        ground_truth, all_atom_mask = get_renamings(
            aa_type, ground_truth, all_atom_mask, prediction)
    else:
        all_atom_mask = jnp.ones(ground_truth.shape, dtype=jnp.bool_).at[:, 4:].set(0)
    same_item = batch[:, None] == batch[None, :]
    pair_mask = all_atom_mask[:, None] * all_atom_mask[None, :] * same_item[..., None]
    pframes, _ = extract_aa_frames(prediction)
    pframes = pframes.to_array()
    pframes = Rigid3Array.from_array(pframes)
    gframes, _ = extract_aa_frames(ground_truth)
    pglobal = pframes[:, None, None].apply_inverse_to_point(prediction[None, :, :])
    gglobal = gframes[:, None, None].apply_inverse_to_point(ground_truth[None, :, :])
    fape = jnp.where(pair_mask, (pglobal - gglobal).norm(), 0)
    fape = fape.sum(axis=-1) / jnp.maximum(pair_mask.sum(axis=-1), 1e-6)
    rotamer_error = jnp.diagonal(fape, 0, -2, -1)
    rotamer_loss = rotamer_error.sum(axis=-1) / jnp.maximum(all_atom_mask.any(axis=-1).sum(axis=-1), 1e-6)
    rotamer_error = rotamer_error[-1]
    pair_mask = pair_mask.any(axis=-1)
    final_ae = fape[-1]
    same_chain = (chain[:, None] == chain[None, :]) * pair_mask
    other_chain = (chain[:, None] != chain[None, :]) * pair_mask
    intra_coin = jnp.zeros_like(jax.random.bernoulli(hk.next_rng_key(), 1 - p_clip, batch.shape))
    inter_coin = jnp.zeros_like(jax.random.bernoulli(hk.next_rng_key(), 1 - p_clip, batch.shape))
    intra_fape = jnp.where(same_chain, jnp.clip(fape, 0.0, jnp.where(intra_coin, jnp.inf, intra_clip)), 0).sum(axis=(-1, -2))
    intra_fape /= jnp.maximum((pair_mask * same_chain).sum(axis=(-1, -2)), 1e-6)
    intra_fape /= intra_clip
    inter_fape = jnp.where(other_chain, jnp.clip(fape, 0.0, jnp.where(inter_coin, jnp.inf, inter_clip)), 0).sum(axis=(-1, -2))
    inter_fape /= jnp.maximum((pair_mask * other_chain).sum(axis=(-1, -2)), 1e-6)
    inter_fape /= inter_clip
    fape = intra_fape + inter_fape
    return fape, rotamer_loss, final_ae, rotamer_error

def alphafold_loss(config, value, batch: BatchType):
    fun = jax.vmap(
        lambda v, b: single_protein_loss(config, v, b),
        in_axes=(0, 0),
        out_axes=0
    )
    result = fun(value, batch)
    result["loss"] *= batch["has_structure"]
    return result

def single_protein_loss(config,
                        value: Mapping[str, Any],
                        batch: Mapping[str, Any],
                        prediction_mask=1
                        ) -> Dict[str, Any]:
  ret = {'loss': 0.}

  ret['metrics'] = {}

  aatype = batch['aatype']
  all_atom_positions = batch['all_atom_positions']
  all_atom_mask = batch['all_atom_mask']
  seq_mask = batch['seq_mask']
  residue_index = batch['residue_index']

  if config.multimer:
    # Split the loss into within-chain and between-chain components.
    intra_chain_mask = batch['asym_id'][:, None] == batch['asym_id'][None, :]
    intra_chain_bb_loss, intra_chain_fape = backbone_loss(
      value["traj"], all_atom_positions, all_atom_mask,
      prediction_mask=intra_chain_mask,
      atom_clamp_distance=config.intra_chain_fape.atom_clamp_distance,
      loss_unit_distance=config.intra_chain_fape.loss_unit_distance
    )
    interface_mask = 1 - intra_chain_mask
    interface_bb_loss, interface_fape = backbone_loss(
      value["traj"], all_atom_positions, all_atom_mask,
      prediction_mask=interface_mask,
      atom_clamp_distance=config.interface_fape.atom_clamp_distance,
      loss_unit_distance=config.interface_fape.loss_unit_distance
    )

    bb_loss = intra_chain_bb_loss + interface_bb_loss
    ret['fape'] = intra_chain_fape + interface_fape
  else:
    # non-multimer bb loss is always intra-chain
    intra_chain_bb_loss, intra_chain_fape = backbone_loss(
      value["traj"], all_atom_positions, all_atom_mask,
      atom_clamp_distance=config.intra_chain_fape.atom_clamp_distance,
      loss_unit_distance=config.intra_chain_fape.loss_unit_distance,
      prediction_mask=prediction_mask
    )

    bb_loss = intra_chain_bb_loss
    ret['fape'] = intra_chain_fape

  ret['bb_loss'] = bb_loss
  ret['loss'] += bb_loss

  if config.predict_sidechains:
    pred_mask = all_atom_multimer.get_atom14_mask(aatype)
    pred_mask *= seq_mask[:, None]
    pred_positions = value['final_atom14_positions']

    sidechains = value['sidechains']
    sc_res = sidechain_loss(
      aatype, sidechains, pred_positions,
      all_atom_positions, all_atom_mask, seq_mask,
      chi_weight=config.sidechain.chi_weight,
      angle_norm_weight=config.sidechain.angle_norm_weight,
      loss_unit_distance=config.loss_unit_distance,
      atom_clamp_distance=config.atom_clamp_distance
    )

    ret['loss'] = ((1 - config.sidechain.weight_frac) * ret['loss'] +
                  config.sidechain.weight_frac * sc_res.sc_loss)
    ret['sidechain_fape'] = sc_res.sc_fape
    ret['loss'] += sc_res.sup_chi_loss

  if config.structural_violation_loss_weight:
    loss, violations = violation_loss(
      aatype, residue_index, pred_positions,
      pred_mask, seq_mask,
      clash_overlap_tolerance=config.clash_overlap_tolerance,
      violation_tolerance_factor=config.violation_tolerance_factor
    )
    ret['metrics'].update(violations)
    ret['loss'] += config.structural_violation_loss_weight * loss

  return ret

def backbone_loss(frames: jnp.ndarray, all_atom_positions: jnp.ndarray, 
                  all_atom_mask: jnp.ndarray, prediction_mask=1,
                  atom_clamp_distance=10.0,
                  loss_unit_distance=20.0):
  all_atom_positions = geometry.Vec3Array.from_array(
    all_atom_positions
  )
  gt_rigid, gt_affine_mask = fmm.make_backbone_affine(
    all_atom_positions, all_atom_mask, None
  )
  gt_affine_mask = gt_affine_mask
  pair_mask = gt_affine_mask[:, None] * gt_affine_mask[None, :]

  if frames.ndim < 3:
    frames = frames[None]
  target_rigid = geometry.Rigid3Array.from_array(frames)
  gt_frames_mask = gt_affine_mask
  bb_loss, fape = backbone_loss_aux(
    gt_rigid=gt_rigid,
    gt_frames_mask=gt_frames_mask,
    gt_positions_mask=gt_affine_mask,
    target_rigid=target_rigid,
    config=dotdict(
      atom_clamp_distance=atom_clamp_distance,
      loss_unit_distance=loss_unit_distance
    ),
    pair_mask=pair_mask * prediction_mask
  )
  return bb_loss, fape

def backbone_loss_aux(gt_rigid: geometry.Rigid3Array,
                      gt_frames_mask: jnp.ndarray,
                      gt_positions_mask: jnp.ndarray,
                      target_rigid: geometry.Rigid3Array,
                      config: dotdict,
                      pair_mask: jnp.ndarray
                      ) -> Tuple[Float, jnp.ndarray]:
  """Backbone FAPE Loss."""
  loss_fn = partial(
    all_atom_multimer.frame_aligned_point_error,
    l1_clamp_distance=config.atom_clamp_distance,
    length_scale=config.loss_unit_distance)

  loss_fn = jax.vmap(loss_fn, (0, None, None, 0, None, None, None))
  fape = loss_fn(target_rigid, gt_rigid, gt_frames_mask,
                 target_rigid.translation, gt_rigid.translation,
                 gt_positions_mask, pair_mask)

  return jnp.mean(fape), fape[-1]

def sidechain_fape(aatype, frames, positions,
                   all_atom_positions, all_atom_mask,
                   gt_positions, gt_mask, alt_naming_is_better,
                   loss_unit_distance=20.0,
                   atom_clamp_distance=10.0):
  pred_frames = geometry.Rigid3Array.from_array(frames)
  pred_positions = geometry.Vec3Array.from_array(positions)
  all_atom_positions = geometry.Vec3Array.from_array(all_atom_positions)
  # print(aatype.shape, all_atom_positions.shape, all_atom_mask.shape, alt_naming_is_better.shape)
  gt_sc_frames, gt_sc_frames_mask = fmm.compute_frames(
    aatype=aatype,
    all_atom_positions=all_atom_positions,
    all_atom_mask=all_atom_mask,
    use_alt=alt_naming_is_better
  )

  sc_loss = fmm.sidechain_loss(
    gt_frames=gt_sc_frames,
    gt_frames_mask=gt_sc_frames_mask,
    gt_positions=gt_positions,
    gt_mask=gt_mask,
    pred_frames=pred_frames,
    pred_positions=pred_positions,
    config=dotdict(sidechain=dotdict(
      loss_unit_distance=loss_unit_distance,
      l1_clamp_distance=atom_clamp_distance
    ))
  )
  return sc_loss["loss"], sc_loss["fape"]

def sidechain_loss(aatype, sidechains, pred_positions,
                   all_atom_positions, all_atom_mask,
                   seq_mask, chi_weight=0.9,
                   angle_norm_weight=0.1,
                   loss_unit_distance=20.0,
                   atom_clamp_distance=10.0,
                   ):
  sc = compute_sidechains(
    pred_positions, aatype,
    all_atom_positions, all_atom_mask
  )

  # print(aatype.shape, sidechains["frames"].shape, sidechains["atom_pos"].shape)
  sc_loss, sc_fape = sidechain_fape(
    aatype, sidechains["frames"], sidechains["atom_pos"],
    all_atom_positions, all_atom_mask,
    sc.gt_positions, sc.gt_mask,
    sc.alt_naming_is_better,
    loss_unit_distance=loss_unit_distance,
    atom_clamp_distance=atom_clamp_distance
  )

  unnormed_angles = sidechains['unnormalized_angles_sin_cos']
  pred_angles = sidechains['angles_sin_cos']

  sup_chi_loss, chi_loss, norm_loss \
    = chi_angle_loss(
      sequence_mask=seq_mask,
      target_chi_mask=sc.gt_chi_mask,
      target_chi_angles=sc.gt_chi_angles,
      aatype=aatype,
      pred_angles=pred_angles,
      unnormed_angles=unnormed_angles,
      config=dotdict(
        chi_weight=chi_weight,
        angle_norm_weight=angle_norm_weight
      )
    )
  return dotdict(
    sup_chi_loss=sup_chi_loss, chi_loss=chi_loss,
    norm_loss=norm_loss, sc_loss=sc_loss, sc_fape=sc_fape
  )

chi_angle_loss = fmm.supervised_chi_loss

def compute_sidechains(pred_positions, aatype,
                       all_atom_positions, all_atom_mask):
  pred_positions = geometry.Vec3Array.from_array(pred_positions)
  all_atom_positions = geometry.Vec3Array.from_array(all_atom_positions)  
  chi_angles, chi_mask = all_atom_multimer.compute_chi_angles(
    all_atom_positions, all_atom_mask, aatype
  )
  gt_positions, gt_mask, alt_naming_is_better = fmm.compute_atom14_gt(
    aatype, all_atom_positions, all_atom_mask, pred_positions
  )
  gt_chi_angles = fmm.get_renamed_chi_angles(
    aatype, chi_angles, alt_naming_is_better
  )
  return dotdict(
    gt_positions=gt_positions,
    gt_mask=gt_mask,
    gt_chi_angles=gt_chi_angles,
    gt_chi_mask=chi_mask,
    alt_naming_is_better=alt_naming_is_better
  )

def violation_loss(aatype, residue_index, pred_positions,
                   pred_mask, seq_mask,
                   clash_overlap_tolerance=...,#TODO
                   violation_tolerance_factor=...,
                   chain_index=None,
                   batch_index=None,
                   per_residue=False): # TODO
  pred_positions = geometry.Vec3Array.from_array(pred_positions)
  violations = find_structural_violations(
    aatype=aatype,
    residue_index=residue_index,
    chain_index=chain_index,
    batch_index=batch_index,
    mask=pred_mask,
    pred_positions=pred_positions,
    config=dotdict(
      clash_overlap_tolerance=clash_overlap_tolerance,
      violation_tolerance_factor=violation_tolerance_factor
    )
  )

  # Several violation metrics:
  violation_metrics = fmm.compute_violation_metrics(
    residue_index=residue_index,
    mask=pred_mask,
    seq_mask=seq_mask,
    pred_positions=pred_positions,
    violations=violations
  )

  if per_residue:
    loss = per_residue_violation_loss(pred_mask, violations, dotdict())
  else:
    loss = custom_structural_violation_loss(
      mask=pred_mask, violations=violations, config=dotdict(
        structural_violation_loss_weight=1.0
      )
    )

  return loss, violation_metrics

def find_structural_violations(
    aatype: jnp.ndarray,
    residue_index: jnp.ndarray,
    chain_index: Any,
    batch_index: Any,
    mask: jnp.ndarray,
    pred_positions: geometry.Vec3Array,  # (N, 14)
    config: dotdict
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
  atom_radius = mask * afutils.batched_gather(atomtype_radius,
                                            residx_atom14_to_atom37)

  # Compute the between residue clash loss.
  between_residue_clashes = custom_clash_loss(
      pred_positions=pred_positions,
      atom_exists=mask,
      atom_radius=atom_radius,
      residue_index=residue_index,
      overlap_tolerance_soft=config.clash_overlap_tolerance,
      overlap_tolerance_hard=config.clash_overlap_tolerance,
      chain_index=chain_index,
      batch_index=batch_index
  )

  # Compute all within-residue violations (clashes,
  # bond length and angle violations).
  restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
      overlap_tolerance=config.clash_overlap_tolerance,
      bond_length_tolerance_factor=config.violation_tolerance_factor)
  dists_lower_bound = afutils.batched_gather(restype_atom14_bounds['lower_bound'],
                                           aatype)
  dists_upper_bound = afutils.batched_gather(restype_atom14_bounds['upper_bound'],
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

# monkey-patch AlphaFold structural violation loss:
def custom_clash_loss(
    pred_positions: geometry.Vec3Array,  # (N, 14)
    atom_exists: jnp.ndarray,  # (N, 14)
    atom_radius: jnp.ndarray,  # (N, 14)
    residue_index: jnp.ndarray,  # (N)
    chain_index: Optional[jnp.ndarray] = None, # (N)
    batch_index: Optional[jnp.ndarray] = None, # (N)
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5) -> Dict[Text, jnp.ndarray]:
  """Loss to penalize steric clashes between residues."""
  assert len(pred_positions.shape) == 2
  assert len(atom_exists.shape) == 2
  assert len(atom_radius.shape) == 2
  assert len(residue_index.shape) == 1

  if chain_index is not None:
    # offset chains, so they don't accidentally run into c_n masking, etc
    residue_index += 2000 * chain_index

  # Create the distance matrix.
  # (N, N, 14, 14)
  dists = geometry.euclidean_distance(pred_positions[:, None, :, None],
                                      pred_positions[None, :, None, :], 1e-10)

  # Create the mask for valid distances.
  # shape (N, N, 14, 14)
  dists_mask = (atom_exists[:, None, :, None] * atom_exists[None, :, None, :])

  # Mask out all the duplicate entries in the lower triangular matrix.
  # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
  # are handled separately.
  dists_mask *= (
      residue_index[:, None, None, None] < residue_index[None, :, None, None])

  # If provided a batch_index: mask out all possible clashes between
  # distinct items in a batch:
  if batch_index is not None:
    dists_mask *= (
        batch_index[:, None, None, None] == batch_index[None, :, None, None])

  # Backbone C--N bond between subsequent residues is no clash.
  c_one_hot = jax.nn.one_hot(2, num_classes=14)
  n_one_hot = jax.nn.one_hot(0, num_classes=14)
  neighbour_mask = ((residue_index[:, None, None, None] +
                     1) == residue_index[None, :, None, None])
  c_n_bonds = neighbour_mask * c_one_hot[None, None, :,
                                         None] * n_one_hot[None, None, None, :]
  dists_mask *= (1. - c_n_bonds)

  # Disulfide bridge between two cysteines is no clash.
  cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
  cys_sg_one_hot = jax.nn.one_hot(cys_sg_idx, num_classes=14)
  disulfide_bonds = (cys_sg_one_hot[None, None, :, None] *
                     cys_sg_one_hot[None, None, None, :])
  dists_mask *= (1. - disulfide_bonds)

  # Compute the lower bound for the allowed distances.
  # shape (N, N, 14, 14)
  dists_lower_bound = dists_mask * (
      atom_radius[:, None, :, None] + atom_radius[None, :, None, :])

  # Compute the error.
  # shape (N, N, 14, 14)
  dists_to_low_error = dists_mask * jax.nn.relu(
      dists_lower_bound - overlap_tolerance_soft - dists)


  # Compute the per atom loss sum.
  # shape (N, 14)
  per_atom_loss_sum = (jnp.sum(dists_to_low_error, axis=[0, 2]) +
                       jnp.sum(dists_to_low_error, axis=[1, 3]))

  # Compute the hard clash mask.
  # shape (N, N, 14, 14)
  clash_mask = dists_mask * (
      dists < (dists_lower_bound - overlap_tolerance_hard))

  # Compute the mean loss.
  # shape ()
  mean_loss = (jnp.sum(dists_to_low_error)
               / (1e-6 + jnp.sum(clash_mask)))

  # Compute the per atom clash.
  # shape (N, 14)
  per_atom_clash_mask = jnp.maximum(
      jnp.max(clash_mask, axis=[0, 2]),
      jnp.max(clash_mask, axis=[1, 3]))

  return {'mean_loss': mean_loss,  # shape ()
          'per_atom_loss_sum': per_atom_loss_sum,  # shape (N, 14)
          'per_atom_clash_mask': per_atom_clash_mask  # shape (N, 14)
         }

setattr(all_atom_multimer, "between_residue_clash_loss", custom_clash_loss)

def custom_structural_violation_loss(mask: jnp.ndarray,
                                     violations: Mapping[str, Float],
                                     config: Any
                                     ) -> Float:
  """Computes Loss for structural Violations."""
  # Put all violation losses together to one large loss.
  between_residues = violations['between_residues']
  within_residues = violations['within_residues']
  num_atoms = jnp.sum(mask).astype(jnp.float32) + 1e-6
  return (config.structural_violation_loss_weight *
          (between_residues['bonds_c_n_loss_mean'] +
           0.3 * between_residues['angles_ca_c_n_loss_mean'] +
           0.3 * between_residues['angles_c_n_ca_loss_mean'] +
           between_residues['clashes_mean_loss'] +
           jnp.sum(within_residues['per_atom_loss_sum']) / num_atoms
           ))

def per_residue_violation_loss(mask: jnp.ndarray,
                               violations: Mapping[str, Float],
                               config: Any) -> Float:
  between_residues = violations['between_residues']
  within_residues = violations['within_residues']
  within_residues_per_residue_loss = within_residues['per_atom_loss_sum'].sum(axis=-1) \
                                   / jnp.maximum(within_residues['per_atom_violations'].sum(axis=-1), 1e-6)
  per_residue_loss = between_residues['connections_per_residue_loss_sum']
  per_residue_loss += (between_residues['clashes_per_atom_loss_sum'] * between_residues['clashes_per_atom_clash_mask']).sum(axis=-1) \
                      / jnp.maximum(between_residues['clashes_per_atom_clash_mask'].sum(axis=-1), 1e-6)
  per_residue_loss += within_residues_per_residue_loss
  per_residue_loss *= mask.any(axis=-1)
  return per_residue_loss

# FIXME: monkeypatch alphafold model utils mask mean
def custom_mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
  """Masked mean."""
  if drop_mask_channel:
    mask = mask[..., 0]

  mask_shape = mask.shape
  value_shape = value.shape

  assert len(mask_shape) == len(value_shape)

  if isinstance(axis, int):
    axis = [axis]
  elif axis is None:
    axis = list(range(len(mask_shape)))
  assert isinstance(axis, (list, tuple)), (
      'axis needs to be either an iterable, integer or "None"')

  broadcast_factor = 1.
  for axis_ in axis:
    value_size = value_shape[axis_]
    mask_size = mask_shape[axis_]
    if mask_size == 1:
      broadcast_factor *= value_size
    else:
      assert mask_size == value_size

  return (jnp.sum(mask * value, axis=axis) /
          (jnp.sum(mask, axis=axis) * broadcast_factor + eps))

setattr(afutils, "mask_mean", custom_mask_mean)
