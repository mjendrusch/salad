from typing import Optional
from functools import partial
from copy import deepcopy

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
    resi_dual, prenorm_skip, resi_dual_input, prenorm_input, drop)

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
    LinearToPoints,
    sequence_relative_position, distance_features,
    direction_features, type_position_features,
    position_rotation_features
)

from salad.modules.utils.diffusion import diffuse_sequence

class S2S(hk.Module):
    def __init__(self, config,
                 name: Optional[str] = "structure_to_sequence"):
        super().__init__(name)
        self.config = config

    def prepare_data(self, data):
        c = self.config
        pos = data["all_atom_positions"]
        atom_mask = data["all_atom_mask"]
        chain = data["chain_index"]
        batch = data["batch_index"]

        # recast positions into atom14:
        # first, truncate to atom14 format
        pos = pos[:, :14]
        atom_mask = atom_mask[:, :14]

        # uniquify chain IDs across batches:
        chain = unique_chain(chain, batch)
        mask = data["seq_mask"] * data["residue_mask"] * atom_mask[:, :3].all(axis=-1)
        
        # subtract the center from all positions
        center = index_mean(pos[:, 1], batch, atom_mask[:, 1, None])
        pos = pos - center[:, None]
        # set the positions of all masked atoms to the pseudo Cb position
        atom_pos = jnp.where(
            atom_mask[..., None], pos,
            compute_pseudo_cb(pos)[:, None, :])

        # transform data to ncacocb positions
        pos = positions_to_ncacocb(atom_pos)

        # add noise to positions
        if not c.eval:
            pos += c.noise_level * jax.random.normal(hk.next_rng_key(), pos.shape)

        # get neighbours
        neighbours = get_nearest_neighbours(c.num_neighbours)(pos, batch, mask)

        result = dict(pos=pos, neighbours=neighbours, chain_index=chain, mask=mask)

        # get SMOL features
        if "smol_positions" in data:
            smol_features = dict(
                pos=data["smol_positions"],
                atom_type=data["smol_types"],
                mask=data["smol_mask"]
            )
            if not c.eval:
                smol_features["pos"] += c.noise_level * jax.random.normal(
                    hk.next_rng_key(), smol_features["pos"].shape)
            result["smol_features"] = smol_features

        return result

    def apply_diffusion(self, data):
        c = self.config
        chain = data["chain_index"]
        aa = data["aa_gt"]
        # use beta-linear task distribution, per chain
        t_beta = jax.random.beta(hk.next_rng_key(), 3, 9, aa.shape)
        t_linear = jax.random.uniform(hk.next_rng_key(), aa.shape)
        t_null = 1
        use_null = jax.random.bernoulli(hk.next_rng_key(), 0.1)
        use_beta = jax.random.bernoulli(hk.next_rng_key(), 0.8)
        t_linear_null = jnp.where(use_null, t_null, t_linear)
        t_seq = jnp.where(use_beta, t_beta, t_linear_null)
        t_seq = t_seq[chain]
        if "t_seq" in data:
            t_seq = data["t_seq"]
        aa, corrupt_aa = diffuse_sequence(hk.next_rng_key(), aa, t_seq, mask_index=20)
        return dict(aa=aa, t_seq=t_seq, corrupt_aa=corrupt_aa)

    def __call__(self, data):
        c = self.config
        model = ADM(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))

        # recycling
        def iteration_body(i, prev):
            result = model(data, prev)
            prev = result["local"]
            return jax.lax.stop_gradient(prev)
        prev = model.init_prev(data)
        if c.recycle:
            if not hk.running_init():
                if c.eval:
                    count = 3
                else:
                    count = jax.random.randint(hk.next_rng_key(), (), 0, 4)
                prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = model(data, prev)
        if c.predict_none:
            result["aa_none"] = model(data, prev, override=20 * jnp.ones_like(data["aa"]))
        total, losses = model.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

class S2SInterfaceAware(hk.Module):
    def __init__(self, config,
                 name: Optional[str] = "structure_to_sequence"):
        super().__init__(name)
        self.config = config

    def prepare_data(self, data):
        c = self.config
        pos = data["all_atom_positions"]
        atom_mask = data["all_atom_mask"]
        chain = data["chain_index"]
        batch = data["batch_index"]

        # recast positions into atom14:
        # first, truncate to atom14 format
        pos = pos[:, :14]
        atom_mask = atom_mask[:, :14]

        # uniquify chain IDs across batches:
        chain = unique_chain(chain, batch)
        mask = data["seq_mask"] * data["residue_mask"] * atom_mask[:, :3].all(axis=-1)
        
        # subtract the center from all positions
        center = index_mean(pos[:, 1], batch, atom_mask[:, 1, None])
        pos = pos - center[:, None]
        # set the positions of all masked atoms to the pseudo Cb position
        atom_pos = jnp.where(
            atom_mask[..., None], pos,
            compute_pseudo_cb(pos)[:, None, :])

        # transform data to ncacocb positions
        pos = positions_to_ncacocb(atom_pos)

        # find and flag interface residues
        other_chain = (chain[:, None] != chain[None, :]) * (batch[:, None] == batch[None, :])
        dist = jnp.linalg.norm(pos[:, None, 4] - pos[None, :, 4], axis=-1)
        dist = jnp.where(other_chain, dist, jnp.inf)
        contact = dist <= 8.0
        interface = contact.any(axis=1)

        # add noise to positions
        if not c.eval:
            pos += c.noise_level * jax.random.normal(hk.next_rng_key(), pos.shape)

        # get neighbours
        neighbours = get_nearest_neighbours(c.num_neighbours)(pos, batch, mask)

        result = dict(pos=pos, interface=interface,
                      neighbours=neighbours, chain_index=chain, mask=mask)

        # get SMOL features
        if "smol_positions" in data:
            smol_features = dict(
                pos=data["smol_positions"],
                atom_type=data["smol_types"],
                mask=data["smol_mask"]
            )
            if not c.eval:
                smol_features["pos"] += c.noise_level * jax.random.normal(
                    hk.next_rng_key(), smol_features["pos"].shape)
            result["smol_features"] = smol_features

        return result

    def apply_diffusion(self, data):
        c = self.config
        chain = data["chain_index"]
        aa = data["aa_gt"]
        # use beta-linear task distribution, per chain
        t_beta = jax.random.beta(hk.next_rng_key(), 3, 9, aa.shape)
        t_linear = jax.random.uniform(hk.next_rng_key(), aa.shape)
        t_null = 1
        # use_null = jax.random.bernoulli(hk.next_rng_key(), 0.1)
        use_beta = jax.random.bernoulli(hk.next_rng_key(), 0.8)
        # t_linear_null = jnp.where(use_null, t_null, t_linear)
        t_seq = jnp.where(use_beta, t_beta, t_linear)
        #t_seq = t_linear
        t_seq = t_seq[chain]
        if "t_seq" in data:
            t_seq = data["t_seq"]
        aa, corrupt_aa = diffuse_sequence(hk.next_rng_key(), aa, t_seq, mask_index=20)
        return dict(aa=aa, t_seq=t_seq, corrupt_aa=corrupt_aa)

    def __call__(self, data):
        c = self.config
        model = ADMInterfaceAware(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))

        result = model(data)
        total, losses = model.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

class S2SEfficient(S2S):
    def __call__(self, data):
        c = self.config
        model = ADMEncoder(c)
        decoder = ADMDecoder(c)
        aa_aux = MLP(
            c.local_size * 2, 20, activation=jax.nn.gelu,
            bias=False, final_init=init_zeros(), name="aa_aux")

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # apply noise to data
        data.update(self.apply_diffusion(data))

        # recycling
        def iteration_body(i, prev):
            local = model(data, prev)
            prev = local
            return jax.lax.stop_gradient(prev)
        prev = model.init_prev(data)
        if not hk.running_init():
            if c.eval:
                count = 0
            else:
                count = jax.random.randint(hk.next_rng_key(), (), 0, 4)
            prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        local = model(data, prev)
        data["local"] = local
        result = dict()
        aa_trunk = jax.nn.log_softmax(aa_aux(local), axis=-1)
        result["aa"] = decoder(data["aa"], aa_trunk, data)
        result["aa_none"] = decoder(20 * jnp.ones_like(data["aa"]), aa_trunk, data)
        result["aa_aux"] = aa_trunk
        corrupt_aa = data["corrupt_aa"]
        result["corrupt_aa"] = corrupt_aa * data["mask"] * (data["aa_gt"] != 20)
        result["aa_gt"] = data["aa_gt"]
        total, losses = decoder.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

class S2SInference(S2S):
    def __init__(self, config,
                 name: Optional[str] = "structure_to_sequence"):
        super().__init__(config, name)

    def apply_diffusion(self, data):
        chain = data["chain_index"]
        aa = data["aa_gt"]
        # use beta-linear task distribution, per chain
        t_beta = jax.random.beta(hk.next_rng_key(), 3, 9, aa.shape)
        t_linear = jax.random.uniform(hk.next_rng_key(), aa.shape)
        t_null = 1
        use_null = jax.random.bernoulli(hk.next_rng_key(), 0.1)
        use_beta = jax.random.bernoulli(hk.next_rng_key(), 0.8)
        t_linear_null = jnp.where(use_null, t_null, t_linear)
        t_seq = jnp.where(use_beta, t_beta, t_linear_null)
        t_seq = t_seq[chain]
        if "t_seq" in data:
            t_seq = data["t_seq"]
        aa, corrupt_aa = diffuse_sequence(hk.next_rng_key(), aa, t_seq, mask_index=20)
        return dict(aa=aa, t_seq=t_seq, corrupt_aa=corrupt_aa)

    def __call__(self, data):
        c = self.config
        model = ADM(c)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # initialize data
        data["aa"] = 20 * jnp.ones_like(data["aa_gt"])
        data["corrupt_aa"] = jnp.ones_like(data["aa_gt"], dtype=jnp.bool_)

        # recycling
        def iteration_body(i, prev):
            result = model(data, prev)
            prev = result["local"]
            return jax.lax.stop_gradient(prev)
        def model_step(data, override=None):
            prev = model.init_prev(data)
            count = c.num_recycle
            prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
            return model(data, prev, override=override)
        def sample_step(i, carry):
            aa, log_p = carry
            result = model_step(data, override=aa)
            logits = result["aa"]
            entropy = (-logits * jax.nn.softmax(logits, axis=-1)).sum(axis=-1)
            entropy = jnp.where(aa == 20, entropy, jnp.inf)
            entropy = jnp.where(data["mask"], entropy, jnp.inf)
            pos = jnp.argmin(entropy, axis=0)
            pos_aa = jax.random.categorical(hk.next_rng_key(), logits[pos] / c.temperature, axis=-1)
            aa = aa.at[pos].set(pos_aa)
            log_p += jnp.where(data["mask"][pos], logits[pos, pos_aa], 0)
            return aa, log_p
        if c.single_step:
            out = model_step(data, override=None)
            logits = out["aa"]
            aatype = jnp.argmax(logits, axis=-1)
            log_p = logits[jnp.arange(data["aa_gt"].shape[0]), data["aa_gt"]]
            log_p = (log_p * data["mask"]).sum()
        else:
            aatype, log_p = hk.fori_loop(0, data["mask"].astype(jnp.int32).sum(), sample_step, (data["aa"], 0.0))
        logits = model(data, model.init_prev(data))["aa"]
        log_p_null = logits[jnp.arange(data["aa_gt"].shape[0]), data["aa_gt"]]
        sequence_null = jnp.argmax(logits, axis=-1)
        perplexity = jnp.exp(-(log_p_null * data["mask"]).sum() / jnp.maximum(data["mask"].sum(), 1))
        recovery = ((aatype == data["aa_gt"]) * data["mask"]).sum() / jnp.maximum(data["mask"].sum(), 1)
        recovery_null = ((sequence_null == data["aa_gt"]) * data["mask"]).sum() / jnp.maximum(data["mask"].sum(), 1)
        
        out = dict(
            aatype=aatype,
            perplexity=perplexity,
            log_p=log_p,
            recovery=recovery,
            recovery_null=recovery_null
        )
        return out
    
class S2SEfficientInference(S2SEfficient):
    def __init__(self, config,
                 name: Optional[str] = "structure_to_sequence"):
        super().__init__(config, name)

    def __call__(self, data):
        c = self.config
        encoder = ADMEncoder(c)
        decoder = ADMDecoder(c)
        aa_aux = MLP(
            c.local_size * 2, 20, activation=jax.nn.gelu,
            bias=False, final_init=init_zeros(), name="aa_aux")

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        # initialize data
        data["aa"] = 20 * jnp.ones_like(data["aa_gt"])
        if "aa_constraint" in data:
            data["aa"] = data["aa_constraint"]
        data["corrupt_aa"] = jnp.ones_like(data["aa_gt"], dtype=jnp.bool_)

        # recycling
        def iteration_body(i, prev):
            prev = encoder(data, prev)
            return jax.lax.stop_gradient(prev)
        def encode(data):
            prev = encoder.init_prev(data)
            count = c.num_recycle
            prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
            return encoder(data, prev)
        local = encode(data)
        data["local"] = local
        aa_trunk = aa_aux(local)
        def sample_step(i, carry):
            aa, log_p = carry
            logits = decoder(aa, aa_trunk, data)
            entropy = (-logits * jax.nn.softmax(logits, axis=-1)).sum(axis=-1)
            entropy = jnp.where(aa == 20, entropy, jnp.inf)
            entropy = jnp.where(data["mask"], entropy, jnp.inf)
            pos = jnp.argmin(entropy, axis=0)
            pos_aa = jax.random.categorical(hk.next_rng_key(), logits[pos] / c.temperature, axis=-1)
            aa = aa.at[pos].set(pos_aa)
            log_p += jnp.where(data["mask"][pos], logits[pos, pos_aa], 0)
            return aa, log_p
        count = (data["mask"] * (data["aa"] == 20)).astype(jnp.int32).sum()
        aatype, log_p = hk.fori_loop(0, count, sample_step, (data["aa"], 0.0))
        logits = decoder(data["aa"], aa_trunk, data)
        log_p_null = logits[jnp.arange(data["aa_gt"].shape[0]), data["aa_gt"]]
        sequence_null = jnp.argmax(logits, axis=-1)
        perplexity = jnp.exp(-(log_p_null * data["mask"]).sum() / jnp.maximum(data["mask"].sum(), 1))
        recovery = ((aatype == data["aa_gt"]) * data["mask"]).sum() / jnp.maximum(data["mask"].sum(), 1)
        recovery_null = ((sequence_null == data["aa_gt"]) * data["mask"]).sum() / jnp.maximum(data["mask"].sum(), 1)

        out = dict(
            aatype=aatype,
            perplexity=perplexity,
            log_p=log_p,
            recovery=recovery,
            recovery_null=recovery_null
        )
        return out

class ADM(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config

    def __call__(self, data, prev, override=None):
        c = self.config
        stack = S2SStack(c, S2SBlock)
        aa_head = MLP(
            c.local_size * 2, 20, activation=jax.nn.gelu,
            bias=False, final_init=init_zeros())
        features = self.prepare_features(data, prev, override=override)
        smol_features = None
        if "smol_features" in data:
            smol_features = data["smol_features"]
        local, pos, neighbours, resi, chain, batch, mask = features
        # enable selective cyclisation of chains
        cyclic_mask = None
        if c.cyclic and "cyclic_mask" in data:
            cyclic_mask = data["cyclic_mask"]
        local = stack(
            local, pos, neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask,
            smol_features=smol_features)
        # predict & handle sequence, losses etc.
        result = dict()
        result["local"] = local
        result["aa"] = jax.nn.log_softmax(aa_head(local), axis=-1)
        corrupt_aa = data["corrupt_aa"]
        result["corrupt_aa"] = corrupt_aa * data["mask"] * (data["aa_gt"] != 20)
        result["aa_gt"] = data["aa_gt"]
        return result

    def init_prev(self, data):
        c = self.config
        return jnp.zeros(
            (data["aa"].shape[0], c.local_size),
            dtype=jnp.float32)

    def prepare_features(self, data, prev, override=None):
        c = self.config
        pos = data["pos"]
        neighbours = data["neighbours"]
        aa = jax.nn.one_hot(data["aa"], 21, axis=-1)
        if override is not None:
            aa = jax.nn.one_hot(override, 21, axis=-1)
        if c.single_step:
            aa = jax.nn.one_hot(20 * jnp.ones_like(data["aa"]), 21, axis=-1)
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        relative_positions = (local_pos[:, :, None] - local_pos[:, None, :])
        relative_distance = distance_rbf(relative_positions.norm(), 0.0, 5.0, 16).reshape(aa.shape[0], -1)
        relative_direction = relative_positions.normalized().to_array().reshape(aa.shape[0], -1)
        local_pos = (local_pos / 5.0).to_array().reshape(aa.shape[0], -1)
        dssp = jax.nn.one_hot(assign_dssp(pos, batch, mask)[0], 3)
        local = Linear(c.local_size, bias=False, initializer="linear")(aa)
        local += Linear(c.local_size, bias=False, initializer="linear")(relative_distance)
        local += Linear(c.local_size, bias=False, initializer="linear")(relative_direction)
        local += Linear(c.local_size, bias=False, initializer="linear")(local_pos)
        local += Linear(c.local_size, bias=False, initializer="linear")(dssp)
        local += Linear(c.local_size, bias=False, initializer="linear")(
            hk.LayerNorm([-1], True, True)(prev))
        local = hk.LayerNorm([-1], True, True)(local)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local)
        local = hk.LayerNorm([-1], True, True)(local)

        return local, pos, neighbours, resi, chain, batch, mask

    def loss(self, data, result):
        c = self.config
        mask = data["mask"]
        losses = dict()
        total = 0.0

        aa_predict_mask = mask * (data["aa_gt"] != 20) * result["corrupt_aa"]
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll
        total = aa_nll

        if "aa_none" in result:
            aa_predict_mask = mask * (data["aa_gt"] != 20)
            aa_nll = -(result["aa_none"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
            aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
            aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
            losses["aa_none"] = aa_nll
            total += aa_nll
        return total, losses

class ADMInterfaceAware(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        if c.persistent_pair:
            stack = S2SPairStack(c, S2SPairBlockSimple)
        else:
            stack = S2SStack(c, S2SBlockSimple)
        aa_head = MLP(
            c.local_size * 2, 20, activation=jax.nn.gelu,
            bias=False, final_init=init_zeros())
        features = self.prepare_features(data)
        smol_features = None
        if "smol_features" in data:
            smol_features = data["smol_features"]
        # local, pos, neighbours, resi, chain, batch, mask = features
        # enable selective cyclisation of chains
        cyclic_mask = None
        if c.cyclic and "cyclic_mask" in data:
            cyclic_mask = data["cyclic_mask"]
        result = stack(
            *features,
            cyclic_mask=cyclic_mask,
            smol_features=smol_features)
        if c.persistent_pair:
            local, pair = result
        else:
            local = result
        # predict & handle sequence, losses etc.
        result = dict()
        result["local"] = local
        result["aa"] = jax.nn.log_softmax(aa_head(local), axis=-1)
        corrupt_aa = data["corrupt_aa"]
        result["corrupt_aa"] = corrupt_aa * data["mask"] * (data["aa_gt"] != 20)
        result["aa_gt"] = data["aa_gt"]
        return result

    def prepare_features(self, data):
        c = self.config
        pos = data["pos"]
        interface = data["interface"]
        neighbours = data["neighbours"]
        aa = jax.nn.one_hot(data["aa"], 21, axis=-1)
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        relative_positions = (local_pos[:, :, None] - local_pos[:, None, :])
        relative_distance = distance_rbf(relative_positions.norm(), 0.0, 5.0, 16).reshape(aa.shape[0], -1)
        relative_direction = relative_positions.normalized().to_array().reshape(aa.shape[0], -1)
        local_pos = (local_pos / 5.0).to_array().reshape(aa.shape[0], -1)
        dssp = jax.nn.one_hot(assign_dssp(pos, batch, mask)[0], 3)
        local = Linear(c.local_size, bias=False, initializer="linear")(aa)
        local += Linear(c.local_size, bias=False, initializer="linear")(relative_distance)
        local += Linear(c.local_size, bias=False, initializer="linear")(relative_direction)
        local += Linear(c.local_size, bias=False, initializer="linear")(local_pos)
        local += Linear(c.local_size, bias=False, initializer="linear")(dssp)
        local += Linear(c.local_size, bias=False, initializer="linear")(interface[:, None])
        local = hk.LayerNorm([-1], True, True)(local)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local)
        local = hk.LayerNorm([-1], True, True)(local)
        if c.persistent_pair:
            pair, _ = pair_features(c)(
                Vec3Array.from_array(pos),
                neighbours, resi, chain, batch, mask)
            pair = hk.LayerNorm([-1], True, True)(pair)
            return local, pair, pos, neighbours, resi, chain, batch, mask

        return local, pos, neighbours, resi, chain, batch, mask

    def loss(self, data, result):
        c = self.config
        mask = data["mask"]
        losses = dict()
        total = 0.0

        aa_predict_mask = mask * (data["aa_gt"] != 20) * result["corrupt_aa"]
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll
        total = aa_nll

        interface_predict_mask = aa_predict_mask * data["interface"] > 0
        inll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        inll = jnp.where(interface_predict_mask, inll, 0)
        inll = inll.sum() / jnp.maximum(interface_predict_mask.sum(), 1)
        losses["interface_aa"] = inll
        total += 2 * inll

        return total, losses

class ADMEncoder(hk.Module):
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config

    def __call__(self, data, prev):
        c = self.config
        stack = S2SStack(c, S2SBlock)
        features = self.prepare_features(data, prev)
        smol_features = None
        if "smol_features" in data:
            smol_features = data["smol_features"]
        local, pos, neighbours, resi, chain, batch, mask = features
        # enable selective cyclisation of chains
        cyclic_mask = None
        if c.cyclic and "cyclic_mask" in data:
            cyclic_mask = data["cyclic_mask"]
        local = stack(
            local, pos, neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask,
            smol_features=smol_features)
        return local

    def init_prev(self, data):
        c = self.config
        return jnp.zeros(
            (data["aa"].shape[0], c.local_size),
            dtype=jnp.float32)

    def prepare_features(self, data, prev):
        c = self.config
        pos = data["pos"]
        neighbours = data["neighbours"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        relative_positions = (local_pos[:, :, None] - local_pos[:, None, :])
        relative_distance = distance_rbf(relative_positions.norm(), 0.0, 5.0, 16).reshape(batch.shape[0], -1)
        relative_direction = relative_positions.normalized().to_array().reshape(batch.shape[0], -1)
        local_pos = (local_pos / 5.0).to_array().reshape(batch.shape[0], -1)
        dssp = jax.nn.one_hot(assign_dssp(pos, batch, mask)[0], 3)
        local = Linear(c.local_size, bias=False, initializer="linear")(relative_distance)
        local += Linear(c.local_size, bias=False, initializer="linear")(relative_direction)
        local += Linear(c.local_size, bias=False, initializer="linear")(local_pos)
        local += Linear(c.local_size, bias=False, initializer="linear")(dssp)
        local += Linear(c.local_size, bias=False, initializer="linear")(
            hk.LayerNorm([-1], True, True)(prev))
        local = hk.LayerNorm([-1], True, True)(local)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local)
        local = hk.LayerNorm([-1], True, True)(local)

        return local, pos, neighbours, resi, chain, batch, mask

class ADMDecoder(hk.Module):
    def __init__(self, config, name: str | None = None):
        super().__init__(name)
        self.config = deepcopy(config)
        self.config.depth = self.config.decoder_depth

    def __call__(self, aa, aa_trunk, data):
        c = self.config
        stack = S2SStack(c, S2SBlock)
        aa_head = MLP(
            c.local_size * 2, 20, activation=jax.nn.gelu,
            bias=False, final_init=init_zeros(), name="aa_head")
        # aa_head = Linear(20, bias=False, initializer=init_zeros(), name="aa_head")
        smol_features = None
        if "smol_features" in data:
            smol_features = data["smol_features"]
        local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(aa, aa_trunk, data)
        local = stack(local, pos, neighbours, resi, chain, batch, mask, smol_features=smol_features)
        local = hk.LayerNorm([-1], True, True)(local)
        aa_logits = jax.nn.log_softmax(aa_head(local), axis=-1)
        return aa_logits

    def loss(self, data, result):
        mask = data["mask"]
        losses = dict()
        total = 0.0

        aa_predict_mask = mask * (data["aa_gt"] != 20) * result["corrupt_aa"]
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll

        aa_predict_mask = mask * (data["aa_gt"] != 20)
        aa_none_nll = -(result["aa_none"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_none_nll = jnp.where(aa_predict_mask, aa_none_nll, 0)
        aa_none_nll = aa_none_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa_none"] = aa_none_nll

        aa_predict_mask = mask * (data["aa_gt"] != 20)
        aa_aux_nll = -(result["aa_aux"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_aux_nll = jnp.where(aa_predict_mask, aa_aux_nll, 0)
        aa_aux_nll = aa_aux_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa_aux"] = aa_aux_nll
        total = aa_nll + aa_none_nll + aa_aux_nll
        return total, losses
    
    def prepare_features(self, aa, aa_trunk, data):
        c = self.config
        pos = data["pos"]
        neighbours = data["neighbours"]
        aa = jax.nn.one_hot(aa, 21, axis=-1)
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        local = data["local"]
        local += Linear(c.local_size, bias=False, initializer="linear")(aa)
        local += Linear(c.local_size, bias=False, initializer="zeros")(
            jax.nn.softmax(aa_trunk, axis=-1))
        return local, pos, neighbours, resi, chain, batch, mask

class S2SStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "s2s_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos, neighbours,
                 resi, chain, batch, mask,
                 smol_features=None,
                 cyclic_mask=None):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features = data
                features = block(c)(
                    features, pos, neighbours,
                    resi, chain, batch, mask,
                    smol_features=smol_features,
                    cyclic_mask=cyclic_mask)
                return features
            return _inner
        diffusion_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=False)(
                hk.remat(stack_inner(diffusion_block)))
        if c.resi_dual:
            # handle ResiDual local features
            incremental = local
            local, incremental = stack(
                ((local, incremental)))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local = stack(local)
            if c.norm_out:
                local = hk.LayerNorm([-1], True, True)(local)
        return local

class S2SPairStack(hk.Module):
    def __init__(self, config, block,
                 name: Optional[str] = "s2s_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pair,
                 pos, neighbours,
                 resi, chain, batch, mask,
                 smol_features=None,
                 cyclic_mask=None):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pair = data
                features, pair = block(c)(
                    features, pair, pos, neighbours,
                    resi, chain, batch, mask,
                    smol_features=smol_features,
                    cyclic_mask=cyclic_mask)
                return features, pair
            return _inner
        diffusion_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=False)(
                hk.remat(stack_inner(diffusion_block)))
        if c.resi_dual:
            # handle ResiDual local features
            incremental = local
            (local, incremental), pair = stack(
                ((local, incremental), pair))
            local = local + hk.LayerNorm([-1], True, True)(incremental)
        else:
            local, pair = stack((local, pair))
            if c.norm_out:
                local = hk.LayerNorm([-1], True, True)(local)
        pair = hk.LayerNorm([-1], True, True)(pair)
        return local, pair

def get_nearest_neighbours(num_neighbours):
    def inner(pos, batch, mask):
        if not isinstance(pos, Vec3Array):
            pos = Vec3Array.from_array(pos)
        same_batch = batch[:, None] == batch[None, :]
        cb = pos[:, 4]
        dist = (cb[:, None] - cb[None, :]).norm()
        dist = jnp.where(same_batch, dist, jnp.inf)
        return get_neighbours(num_neighbours)(dist, mask)
    return inner

def pair_features(c):
    def inner(pos, neighbours, resi, chain, batch, mask, cyclic_mask=None):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, cyclic=cyclic_mask is not None)(
                resi, chain, batch, neighbours, cyclic_mask=cyclic_mask))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

class S2SBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "s2s_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, neighbours,
                 resi, chain, batch, mask,
                 smol_features=None,
                 cyclic_mask=None):
        c = self.config
        def do_drop(x):
            if c.drop:
                return drop(x, p=0.1, is_training=not c.eval)
            return x
        residual_update, residual_input, _ = get_residual_gadgets(
            features, c.resi_dual)

        attention = SparseStructureAttention

        pair, pair_mask = pair_features(c)(
            Vec3Array.from_array(pos), neighbours,
            resi, chain, batch, mask,
            cyclic_mask=cyclic_mask)
        pair = do_drop(pair)
        features = residual_update(
            features,
            do_drop(attention(c)(
                residual_input(features), pos / c.sigma_data, pair, pair_mask,
                neighbours, resi, chain, batch, mask)))
        if smol_features is not None:
            local_update = SMOLUpdate(c)(
                residual_input(features),
                pos,
                smol_features["atom_type"],
                smol_features["pos"],
                smol_features["mask"])
            features = residual_update(features, do_drop(local_update))
        # global update of local features
        features = residual_update(
            features,
            do_drop(Update(c)(
                residual_input(features),
                pos, chain, batch, mask,
                neighbours)))
        return features

class S2SPairBlock(hk.Module):
    def __init__(self, config, name: Optional[str] = "s2s_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pair, pos, neighbours,
                 resi, chain, batch, mask,
                 smol_features=None,
                 cyclic_mask=None):
        c = self.config
        def do_drop(x):
            if c.drop:
                return drop(x, p=0.1, is_training=not c.eval)
            return x
        residual_update, residual_input, _ = get_residual_gadgets(
            features, c.resi_dual)

        attention = SparseStructureAttention

        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair += do_drop(LocalToPair(c)(
            residual_input(features),
            hk.LayerNorm([-1], True, True)(pair),
            neighbours))
        pair += do_drop(PairToPair(c)(
            hk.LayerNorm([-1], True, True)(pair),
            pos, neighbours, pair_mask))
        features = residual_update(
            features,
            do_drop(attention(c)(
                residual_input(features), pos / c.sigma_data,
                hk.LayerNorm([-1], True, True)(pair), pair_mask,
                neighbours, resi, chain, batch, mask)))
        if smol_features is not None:
            local_update = SMOLUpdate(c)(
                residual_input(features),
                pos,
                smol_features["atom_type"],
                smol_features["pos"],
                smol_features["mask"])
            features = residual_update(features, do_drop(local_update))
        # global update of local features
        features = residual_update(
            features,
            do_drop(Update(c)(
                residual_input(features),
                pos, chain, batch, mask,
                neighbours)))
        return features, pair
    
class S2SBlockSimple(hk.Module):
    def __init__(self, config, name: Optional[str] = "s2s_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos, neighbours,
                 resi, chain, batch, mask,
                 smol_features=None,
                 cyclic_mask=None):
        c = self.config
        def do_drop(x):
            if c.drop:
                return drop(x, p=0.1, is_training=not c.eval)
            return x
        residual_update, residual_input, _ = get_residual_gadgets(
            features, c.resi_dual)

        attention = SparseStructureAttention

        pair, pair_mask = pair_features(c)(
            Vec3Array.from_array(pos),
            neighbours, resi, chain, batch, mask,
            cyclic_mask=cyclic_mask)
        features = residual_update(
            features,
            do_drop(attention(c)(
                residual_input(features), pos / c.sigma_data, do_drop(pair), pair_mask,
                neighbours, resi, chain, batch, mask)))
        if smol_features is not None:
            local_update = SMOLUpdate(c)(
                residual_input(features),
                pos,
                smol_features["atom_type"],
                smol_features["pos"],
                smol_features["mask"])
            features = residual_update(features, do_drop(local_update))
        # GeGLU MLP update
        features = residual_update(
            features,
            do_drop(SimpleUpdate(c)(
                residual_input(features))))
        return features

class S2SPairBlockSimple(hk.Module):
    def __init__(self, config, name: Optional[str] = "s2s_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pair, pos, neighbours,
                 resi, chain, batch, mask,
                 smol_features=None,
                 cyclic_mask=None):
        c = self.config
        def do_drop(x):
            if c.drop:
                return drop(x, p=0.1, is_training=not c.eval)
            return x
        residual_update, residual_input, _ = get_residual_gadgets(
            features, c.resi_dual)

        attention = SparseStructureAttention

        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair += LocalToPair(c)(
            residual_input(features),
            hk.LayerNorm([-1], True, True)(pair),
            neighbours)
        pair += PairToPair(c)(
            hk.LayerNorm([-1], True, True)(pair),
            pos, neighbours, pair_mask)
        features = residual_update(
            features,
            do_drop(attention(c)(
                residual_input(features), pos / c.sigma_data, do_drop(pair), pair_mask,
                neighbours, resi, chain, batch, mask)))
        if smol_features is not None:
            local_update = SMOLUpdate(c)(
                residual_input(features),
                pos,
                smol_features["atom_type"],
                smol_features["pos"],
                smol_features["mask"])
            features = residual_update(features, do_drop(local_update))
        # GeGLU MLP update
        features = residual_update(
            features,
            do_drop(SimpleUpdate(c)(
                residual_input(features))))
        return features, pair

class LocalToPair(hk.Module):
    def __init__(self, config, name: str | None = None):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, neighbours):
        outer_left = jax.nn.gelu(Linear(8, bias=False)(local))
        outer_right = Linear(8, bias=False)(local)
        outer = outer_left[:, None, :, None] * outer_right[neighbours, None, :]
        outer = outer.reshape(*neighbours.shape, -1)
        additive_left = Linear(pair.shape[-1], bias=False)(local)
        additive_right = Linear(pair.shape[-1], bias=False)(local)
        additive = additive_left[:, None] + additive_right[neighbours]
        features = jnp.concatenate((pair, outer, additive), axis=-1)
        gate = jax.nn.gelu(Linear(pair.shape[-1] * 2, bias=False)(features))
        hidden = gate * Linear(pair.shape[-1] * 2, bias=False)(features)
        return Linear(pair.shape[-1], bias=False, initializer="zeros")(hidden)

class PairToPair(hk.Module):
    def __init__(self, config, name: str | None = None):
        super().__init__(name)
        self.config = config

    def __call__(self, pair, pos, neighbours, pair_mask):
        c = self.config
        pos = Vec3Array.from_array(pos)
        frames, _ = extract_aa_frames(pos)
        neighbour_pos: Vec3Array = frames[:, None, None].apply_inverse_to_point(pos[neighbours])
        dscale = hk.get_parameter("d_scale", (),
                                  init=hk.initializers.Constant(
                                      jax.nn.softplus(jnp.log(jnp.exp(1.) - 1.))
                                  ))
        dscale = jax.nn.softplus(dscale) / 10.0
        pos_features = dscale * neighbour_pos.to_array().reshape(*neighbours.shape, -1)
        dir_features = neighbour_pos.normalized().to_array().reshape(*neighbours.shape, -1)
        pair_features = jnp.concatenate((pair, pos_features, dir_features), axis=-1)
        left = Linear(pair.shape[-1], bias=False)(pair_features)
        right = Linear(pair.shape[-1], bias=False)(pair_features)
        pair_pair = left[:, :, None] + right[:, None, :8]
        rel: Vec3Array = neighbour_pos[:, :, None, :, None] - neighbour_pos[:, None, :8, None, :]
        dist = rel[:, :, 1, 1].norm()
        dist = distance_rbf(dist, 0.0, 10.0, 16)
        rel_features = dscale * rel.to_array().reshape(*neighbours.shape, 8, -1)
        dirs = rel.normalized().to_array().reshape(*neighbours.shape, 8, -1)
        pair_pair_features = jnp.concatenate((pair_pair, rel_features, dirs), axis=-1)
        gate = jax.nn.gelu(Linear(pair.shape[-1], bias=False)(pair_pair_features))
        hidden = gate * Linear(pair.shape[-1], bias=False)(pair_pair_features)
        pair_pair_mask = pair_mask[:, :, None] * pair_mask[:, None, :8]
        result = jnp.where(
            pair_pair_mask[..., None],
            Linear(pair.shape[-1], bias=False, initializer="zeros")(hidden),
            0).sum(axis=2) / jnp.maximum(pair_pair_mask[..., None].sum(axis=2), 1)
        return result

def get_residual_gadgets(features, use_resi_dual=True):
    residual_update = (lambda x, y: resi_dual(*x, y)) if use_resi_dual else prenorm_skip
    residual_input = resi_dual_input if use_resi_dual else prenorm_input
    local_shape = features[0].shape[-1] if use_resi_dual else features.shape[-1]
    return residual_update, residual_input, local_shape

def make_pair_mask(mask, neighbours):
    return mask[:, None] * mask[neighbours] * (neighbours != -1)

class SMOLUpdate(hk.Module):
    def __init__(self, config, name: Optional[str] = "smol_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, smol_type, smol_pos, smol_mask):
        c = self.config
        frames, _ = extract_aa_frames(Vec3Array.from_array(pos))
        pseudo_pos = LinearToPoints(11)(local, frames)
        pos = jnp.concatenate((pos, pseudo_pos), axis=-2)
        relative = smol_pos[:, :, None, :] - pos[:, None, :, :]
        relative = Vec3Array.from_array(relative)
        directions = relative.normalized().to_array().reshape(*smol_pos.shape[:-1], -1)
        distances = distance_rbf(relative.norm(), 0.0, 12.0, 16).reshape(*smol_pos.shape[:-1], -1)
        pair_mask = smol_mask
        pair = Linear(c.pair_size, bias=False)(
            jax.nn.one_hot(smol_type, 7))
        pair += Linear(c.pair_size, bias=False)(local)[:, None]
        pair += Linear(c.pair_size, bias=False)(directions)
        pair += Linear(c.pair_size, bias=False)(distances)
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(c.pair_size * 2, c.pair_size, activation=jax.nn.gelu)(pair)
        pair = jnp.where(pair_mask[..., None], pair, 0)
        pair = pair.sum(axis=1) / jnp.maximum(pair_mask[..., None].sum(axis=1), 1)
        gate = Linear(pair.shape[-1], bias=False, initializer="relu")(local)
        pair = jax.nn.gelu(gate) * pair
        return Linear(local.shape[-1], bias=False, initializer="zeros")(pair)

class Update(hk.Module):
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, chain, batch, mask, neighbours):
        c = self.config
        type_features = MLP(local.shape[-1] * 2, local.shape[-1],
                            activation=jax.nn.gelu,
                            bias=False, final_init="zeros")(
            type_position_features(local, Vec3Array.from_array(pos), batch, mask,
                                   size=16, learned_offset=True,
                                   neighbours=neighbours))
        type_features = jnp.where(mask[..., None], type_features, 0)
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        chain_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        batch_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        hidden = batch_gate * index_mean(local_update, batch, mask[..., None])
        hidden += chain_gate * index_mean(local_update, chain, mask[..., None])
        hidden += local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(hidden) + type_features
        return result

class SimpleUpdate(hk.Module):
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local):
        c = self.config
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        hidden = local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(hidden)
        return result
