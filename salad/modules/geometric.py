"""This module contains neural network blocks for working with
point clouds and protein chains."""

from typing import Optional, Union, Any

from salad.aflib.model.geometry import Vec3Array, Rigid3Array

import jax
import jax.numpy as jnp

import haiku as hk

from salad.aflib.model.all_atom_multimer import make_transform_from_reference

from salad.modules.basic import Linear, MLP, init_glorot, init_linear
from salad.modules.transformer import Transition
from salad.modules.utils.geometry import (
    distance_rbf, extract_aa_frames, extract_neighbours,
    sequence_relative_position, get_neighbours,
    axis_index, index_mean
)

def distance_features(pos, neighbours=None, d_min=0.0, d_max=22.0, num_rbf=16):
    """Compute residue pair distance features.
    
    Args:
        pos: residue atom positions of shape (N, M, 3).
        neighbours: optional neighbour array.
        d_min: minimum distance for binning. Default: 0.0.
        d_max: maximum distance for binning. Default: 22.0.
        num_rbf: number of radial basis functions.
    Returns:
        5 * 5 * 16 radial basis functions per residue pair.
        If a neighbour array was provided, the output has shape (N, K, 400)
        otherwise it has shape (N, N, 400).
    """
    if neighbours is None:
        dist = (pos[:, None, :5, None] - pos[None, :, None, :5]).norm()
    else:
        dist = (pos[:, None, :5, None] - pos[neighbours, None, :5]).norm()
    dist = dist.reshape(*dist.shape[:-2], -1)
    result = distance_rbf(dist, d_min, d_max, num_rbf).reshape(*dist.shape[:-1], -1)
    return result

def direction_features(pos, neighbours=None, d_min=0.0, d_max=22.0, num_rbf=16):
    """Computes residue relative direction features.
    
    Args:
        pos: residue atom positions of shape (N, atoms, 3).
        neighbours: optional neighbour array.
    Returns:
        Direction vectors from the center of each residue frame to
        all atoms in each of its neighbours (N, K, atoms, 3).
    """
    frames, _ = extract_aa_frames(pos)
    if neighbours is None:
        local_pos = frames[:, None, None].apply_inverse_to_point(pos[None, :])
        dirs = local_pos.normalized().to_array()
    else:
        local_pos = frames[:, None, None].apply_inverse_to_point(pos[neighbours])
        dirs = local_pos.normalized().to_array()
    result = dirs.reshape(*dirs.shape[:2], -1)
    return result

def type_position_features(local, pos, batch, mask, size=32, scale=10.0,
                           learned_offset=False, neighbours=None):
    frames, _ = extract_aa_frames(pos)
    pair_mask = (neighbours != -1) * mask[:, None]
    def type_features(type_pos):
        type_pos = Vec3Array.from_array(type_pos)
        if len(type_pos.shape) == 3:
            type_pos = frames[:, None, None].apply_inverse_to_point(type_pos)
        else:
            type_pos = frames[:, None].apply_inverse_to_point(type_pos)
        type_dir = type_pos.normalized().to_array().reshape(local.shape[0], size * 3)
        type_dist = distance_rbf(type_pos.norm(), 0.0, 22.0, 10).reshape(type_pos.shape[0], -1)
        type_pos = type_pos.to_array().reshape(type_pos.shape[0], size * 3) / scale
        return jnp.concatenate((type_dir, type_dist, type_pos), axis=-1)
    # compute type weight
    base_type_weight = Linear(size, bias=False, initializer="linear")(local)
    base_type_weight = jax.nn.gelu(base_type_weight)
    # local type positions
    type_weight = base_type_weight[neighbours]
    type_weight = jnp.where((neighbours != -1)[..., None], type_weight, 0)
    pos = pos.to_array()
    entry_pos = pos[:, 1, None]
    if learned_offset:
        entry_pos = LinearToPoints(size, init="zeros")(local, frames)
    local_type_pos = entry_pos[neighbours] * type_weight[..., None]
    local_type_pos = jnp.where(pair_mask[..., None, None], local_type_pos, 0)
    local_type_pos = local_type_pos.sum(axis=1) / jnp.maximum(pair_mask.sum(axis=-1)[..., None, None], 1)
    # global type positions
    global_type_pos = index_mean(
        entry_pos * base_type_weight[..., None],
        batch, mask[:, None, None])
    return jnp.concatenate((type_features(local_type_pos),
                            type_features(global_type_pos)), axis=-1)

def paired_distance_features(x, y, d_min=0.0, d_max=22.0, num_rbf=16):
    """Compute distance features for a pair of structures x and y.
    
    Args:
        x, y: residue atom positions of shape (N, atoms, 3).
        d_min: minimum distance for binning. Default: 0.0.
        d_max: maximum distance for binning. Default: 22.0.
        num_rbf: number of radial basis functions.
    Returns:
        5 * 5 * 16 radial basis functions per residue pair.
    """
    dist = (x[..., :, None, :] - y[..., None, :, :]).norm()
    dist = dist.reshape(*dist.shape[:-2], -1)
    return distance_rbf(dist, d_min, d_max, num_rbf).reshape(*dist.shape[:-1], -1)

def rotation_features(frames, neighbours=None):
    """Relative rotation features for a set of residue frames and neighbours.
    
    Args:
        frames (Rigid3Array): local coordinate frames for a set of residues.
        neighbours (Optional[array]): neighbours of each residue
    Returns:
        Entries of the relative rotation matrix between
        pairs of neighbouring residues (N, K, 9).
    """
    if neighbours is None:
        rot = frames[:, None].inverse().rotation @ frames[None, :].rotation
    else:
        rot = frames[:, None].inverse().rotation @ frames[neighbours].rotation
    rot = rot.to_array().reshape(*rot.shape, -1)
    return rot

def position_rotation_features(pos: Vec3Array, neighbours=None):
    """Relative cotation features computed from atom positions.
    
    Args:
        pos (Vec3Array): residue backbone atom coordinates.
        neighbours (Optional[array]): neighbours of each residue
    Returns:
        Entries of the relative rotation matrix between
        pairs of neighbouring residues (N, K, 9).
    """
    frames, _ = extract_aa_frames(pos)
    if neighbours is None:
        rot = frames[:, None].inverse().rotation @ frames[None, :].rotation
    else:
        rot = frames[:, None].inverse().rotation @ frames[neighbours].rotation
    rot = rot.to_array().reshape(*rot.shape, -1)
    return rot

def paired_rotation_features(x, y):
    """Compute distance features for a pair of structures x and y.
    
    Args:
        x, y: residue atom positions of shape (N, atoms, 3).
        neighbours (Optional[array]): neighbours of each residue
    Returns:
        Entries of the relative rotation matrix between
        pairs of residues in x and y.
    """
    x_frames = make_transform_from_reference(x[..., 0], x[..., 1], x[..., 2])
    y_frames = make_transform_from_reference(y[..., 0], y[..., 1], y[..., 2])
    rot = x_frames.inverse().rotation[:, None] @ y_frames.rotation[None, :]
    return rot.to_array().reshape(*rot.shape, -1)

def pair_vector_features(pos, neighbours=None, scale=0.1):
    """Compute local coordinate features.
    
    Args:
        pos: residue atom positions of shape (N, atoms, 3).
        neighbours (Optional[array]): neighbours of each residue
        scale: position scale factor. Default: 0.1.
    Returns:
        Coordinate features of atoms of a residue and its neighbours
        in the residue's local frame.
    """
    if neighbours is None:
        neighbours = jnp.broadcast_to(
            jnp.arange(pos.shape[0], dtype=jnp.int32)[None, :],
            (pos.shape[0], pos.shape[0]))
    frames, _ = extract_aa_frames(pos)
    pair_vectors = jnp.concatenate((
        jnp.broadcast_to(
            frames[:, None, None].apply_inverse_to_point(
                pos[:, None]).to_array(),
            (neighbours.shape[0], neighbours.shape[1], pos.shape[1], 3)),
        frames[:, None, None].apply_inverse_to_point(pos[neighbours]).to_array(),
    ), axis=-2)
    pair_vectors = Vec3Array.from_array(pair_vectors)
    direction = pair_vectors.normalized().to_array().reshape(
        *pair_vectors.shape[:-1], -1)
    result = jnp.concatenate((
        direction,
        scale * pair_vectors.to_array().reshape(
            *pair_vectors.shape[:-1], -1)
    ), axis=-1)
    return result

def frame_pair_features(frames, pos, neighbours, d_min=0.0, d_max=22.0, num_rbf=16):
    """DEPRECATED. Compute frame-derived pair features."""
    pair_features = []
    # distance features:
    dist = (pos[:, None, :5, None] - pos[neighbours, None, :5]).norm()
    dist = dist.reshape(*dist.shape[:-2], -1)
    pair_features.append(
        distance_rbf(dist, d_min, d_max, num_rbf).reshape(
            *dist.shape[:-1], -1))
    # relative rotation features:
    rot = frames[:, None].inverse().rotation @ frames[neighbours].rotation
    rot = rot.to_array().reshape(*rot.shape, -1)
    pair_features.append(rot)
    return jnp.concatenate(pair_features, axis=-1)

def frame_pair_vector_features(frames, pos, neighbours, scale=0.1):
    """DEPRECATED. Compute frame-derived pair features."""
    local_pos = frames[..., None].apply_inverse_to_point(pos)
    local_neighbours = frames[..., None, None].apply_inverse_to_point(
        pos[neighbours])
    combined_pos = jnp.concatenate((
        local_pos.to_array()[:, None].repeat(neighbours.shape[1], axis=1),
        local_neighbours.to_array()), axis=-2)
    combined_dist = jnp.sqrt(jnp.maximum((combined_pos ** 2).sum(axis=-1), 1e-6))
    features = jnp.concatenate((
        scale * combined_dist,
        scale * combined_pos.reshape(*combined_pos.shape[:-2], -1),
        distance_rbf(combined_dist, 0.0, 22.0, 16).reshape(
            *combined_dist.shape[:-1], -1)
    ), axis=-1)
    return features

class SparsePairUpdate(hk.Module):
    """Sparse pair feature update."""
    def __init__(self, config, name: str | None = "sparse_pair_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pair, pair_update, neighbours, mask):
        c = self.config
        pair_update += Linear(pair.shape[-1], bias=False)(local[:, None])
        pair_update += Linear(pair.shape[-1], bias=False)(local[None, :])
        pair_norm = hk.LayerNorm([-1], True, True)
        index = axis_index(neighbours, axis=0)
        local_pair = pair[index[:, None], neighbours]
        local_pair = pair_norm(local_pair)
        augment = Linear(local_pair.shape[-1], initializer="zeros", bias=False)(pair_update)
        local_pair += augment
        linear = Linear(local_pair.shape[-1], initializer="zeros", bias=False)(local_pair)
        left = Linear(32)(local_pair)
        right = Linear(32)(local_pair)
        neighbour_pair = left[:, None, :] + right[neighbours]

        pair_neighbours = neighbours[neighbours]
        pair_neighbours = jnp.where((neighbours == -1)[:, :, None], pair_neighbours, -1)
        pair_mask = (pair_neighbours != -1) * mask[pair_neighbours]
        pair_mask *= (neighbours != -1)[:, None, :]

        pair_update = MLP(pair.shape[-1] * 2, pair.shape[-1], final_init="zeros", bias=False)(neighbour_pair)
        pair_update = jnp.where(pair_mask, pair_update, 0).sum(axis=1)
        pair_update /= jnp.maximum(pair_mask.sum(axis=1), 1)

        interaction = Linear(pair.shape[-1])(local)

        pair = pair.at[index[:, None], neighbours].add(pair_update + interaction + linear)
        return pair

class LinearToPoints(hk.Module):
    """Linear layer returning points in 3D space."""
    def __init__(self, size=8, init="linear", name: Optional[str] = "point_linear"):
        super().__init__(name=name)
        self.size = size
        self.init = init

    def __call__(self, data: jnp.ndarray, frames: Rigid3Array):
        # project input data
        raw = Linear(self.size * 3, bias=False, initializer=self.init)(data).astype(jnp.float32)
        # reshape into an array of 3D vectors
        points = Vec3Array.from_array(
            raw.reshape(*raw.shape[:-1], self.size, 3)
        )
        # project into global frame using local residue frames
        points = frames[..., None].apply_to_point(points)
        return points.to_array().astype(data.dtype)

class VectorLinear(Linear):
    """Equivariant vector-to-vector linear layer."""
    def __init__(self, size: int = 128,
                 initializer: Union[str, hk.initializers.Initializer] = "zeros",
                 name: Optional[str] = "vector_linear"):
        super().__init__(size, False, 0, initializer, name)

    def __call__(self, vec: Union[Vec3Array, jnp.ndarray]) -> Union[Vec3Array, jnp.ndarray]:
        result = vec
        if isinstance(vec, Vec3Array):
            result = vec.to_array()
        # turn an array of 3D vectors into 3 arrays of numbers
        # apply a linear layer without bias
        # turn the result back into an array of 3D vectors
        # this operation is a linear combination of the input vectors
        # and therefore E(3) equivariant.
        result = jnp.swapaxes(super().__call__(jnp.swapaxes(result, -1, -2)), -1, -2)
        if isinstance(vec, Vec3Array):
            result = Vec3Array.from_array(result)
        return result

class VectorMLP(MLP):
    """Equivariant vector-to-vector MLP."""
    def __init__(self, size: int = 64,
                 out_size: int = None,
                 depth: int = 2,
                 activation = jax.nn.relu,
                 init = "relu",
                 final_init = "linear",
                 name: Optional[str] = "vector_mlp"):
        super().__init__(
            size, out_size, depth, activation,
            init, final_init, name)

    def __call__(self, scalar: jnp.ndarray, vector: Vec3Array):
        out_size = self.out_size or vector.shape[-1]
        new_vector = vector
        for _ in range(self.depth - 1):
            # scale the current vector features using a gate
            # computed from the current scalar features
            # and apply a linear layer.
            scale = jax.nn.gelu(Linear(new_vector.shape[-1])(scalar))
            new_vector = VectorLinear(
                self.size, initializer=init_glorot())(new_vector * scale)
            # compute a direction vector from the current vector features
            direction = VectorLinear(
                1, initializer=init_glorot())(new_vector).normalized()
            # use it to apply an equivariant directional nonlinearity
            new_vector = vector_nonlinearity(
                new_vector, direction, activation=jax.nn.gelu)
        # scale the output by a gate computed from scalar features
        # and apply a final linear layer.
        scale = jax.nn.gelu(Linear(new_vector.shape[-1])(scalar))
        new_vector = VectorLinear(
            out_size, initializer=self.final_init)(new_vector * scale)
        return new_vector

class VectorLayerNorm(hk.Module):
    """Equivariant vector layer norm."""
    def __init__(self, name: Optional[str] = "vector_layer_norm"):
        super().__init__(name)

    def __call__(self, vec: Vec3Array) -> Vec3Array:
        norm = vec.norm()
        var = jnp.maximum(((norm - norm.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1, keepdims=True), 1e-6)
        vec = vec * jax.lax.rsqrt(var)
        scale = hk.get_parameter("scale", (vec.shape[-1],), init=hk.initializers.Constant(1.0))
        return scale * vec

def vector_std_norm(vec: Vec3Array) -> Vec3Array:
    """Normalize an array of vectors by its standard deviation."""
    norm = vec.norm()
    var = jnp.maximum(((norm - norm.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1, keepdims=True), 1e-6)
    vec = vec * jax.lax.rsqrt(var)
    return vec

def vector_mean_norm(vec: Vec3Array) -> Vec3Array:
    """Normalize an array of vectors by the mean of their norm."""
    vec = vec.to_array()
    vec = vec - vec.mean(axis=-2, keepdims=True)
    vec /= jax.lax.sqrt(
        jnp.maximum((vec ** 2).sum(axis=-1, keepdims=True), 1e-6)).mean(
            axis=-2, keepdims=True)
    return Vec3Array.from_array(vec)

def vector_nonlinearity(vector: Vec3Array, direction: Vec3Array,
                        activation=jax.nn.relu):
    """Apply an equivariant nonlinearity in a direction."""
    agreement = direction.dot(vector)
    scale = activation(agreement)
    return scale * vector

class SparseStructureMessage(hk.Module):
    """Message passing wrapper."""
    def __init__(self, config,
                 name: Optional[str] = "sparse_structure_message"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, pair, pair_mask, neighbours,
                 resi, chain, batch, mask):
        c = self.config
        pos = Vec3Array.from_array(pos.astype(jnp.float32))
        pair = MLP(
            2 * c.pair_size, local.shape[-1], depth=3,
            activation=jax.nn.gelu, final_init="zeros")(pair)
        local_update = jnp.where(
            pair_mask[..., None], pair, 0).sum(axis=1)
        local_update /= pair.shape[1]
        return local_update

class SparseStructureAttention(hk.Module):
    """Sparse attention wrapper."""
    def __init__(self, config, normalize=True,
                 name: Optional[str] = "sparse_structure_attn"):
        super().__init__(name)
        self.normalize = normalize
        self.config = config

    def __call__(self, local, pos, pair, pair_mask, neighbours,
                 resi, chain, batch, mask):
        c = self.config
        final_init = c.update_init if c.update_init else "zeros"
        frames, _ = extract_aa_frames(Vec3Array.from_array(pos))
        if c.multi_query:
            local_update = SparseInvariantMultiQueryAttention(
                heads=c.heads, size=c.key_size,
                final_init=final_init, normalize=self.normalize)(
                local, pair, frames.to_array(),
                neighbours, pair_mask)
        else:
            local_update = SparseInvariantPointAttention(
                heads=c.heads, size=c.key_size,
                final_init=final_init, normalize=self.normalize)(
                local, pair, frames.to_array(),
                neighbours, pair_mask)
        return local_update

class SemiEquivariantSparseStructureAttention(hk.Module):
    """Sparse attention wrapper."""
    def __init__(self, config, normalize=False,
                 name: Optional[str] = "se_sparse_structure_attn"):
        super().__init__(name)
        self.normalize = normalize
        self.config = config

    def __call__(self, local, pos, pair, pair_mask, neighbours,
                 resi, chain, batch, mask):
        c = self.config
        final_init = c.update_init if c.update_init else "zeros"
        local_update = SparseSemiEquivariantPointAttention(
            heads=c.heads, size=c.key_size,
            final_init=final_init, normalize=self.normalize)(
            local, pair, pos,
            neighbours, pair_mask)
        return local_update

class SparseInvariantMultiQueryAttention(hk.Module):
    """Sparse IPA with multi-query attention."""
    def __init__(self, size=32, heads=4,
                 query_points=8, value_points=8,
                 final_init="zeros", normalize=False,
                 name: Optional[str]="ada_point_attention"):
        super().__init__(name=name)
        self.size = size
        self.heads = heads
        self.query_points = query_points
        self.value_points = value_points
        self.final_init = final_init
        self.normalize = normalize

    def __call__(self, local, pair, frames, neighbours, mask):
        frames: Rigid3Array = Rigid3Array.from_array(frames.astype(jnp.float32))
        def attention_component(x, heads=self.heads):
            result = Linear(heads * self.size, initializer="linear")(x)
            return result.reshape(result.shape[0], heads, self.size)
        def attention_point(x, heads=self.heads):
            result = Linear(heads * self.query_points * 3, initializer="linear")(x)
            result = result.reshape(result.shape[0], heads, self.query_points, 3)
            result = Vec3Array.from_array(result)
            result = frames[:, None, None].apply_to_point(result)
            return result
        def to_local(x: jnp.ndarray):
            x = Vec3Array.from_array(x)
            x = frames[:, None, None].apply_inverse_to_point(x)
            return x.to_array()
        # set up multi-query attention
        q = attention_component(local, heads=self.heads)
        k = attention_component(local, heads=1)
        q = hk.LayerNorm([-1], False, False)(q)
        k = hk.LayerNorm([-1], False, False)(k)
        v = attention_component(local, heads=1)
        qp = attention_point(local, heads=self.heads)
        kp = attention_point(local, heads=1)
        vp = attention_point(local, heads=1)

        # attention matrix
        w_C = jnp.sqrt(2 / (9 * self.query_points))
        w_L = jnp.sqrt(1 / 3)

        gamma = hk.get_parameter(
            "gamma", (self.heads,),
            init=hk.initializers.Constant(jnp.log(jnp.exp(1.) - 1.))
        )
        scale = jax.nn.softplus(gamma.reshape(1, 1, self.heads)) * w_C / 2
        attn_logits = jnp.einsum("ihc,ijhc->ijh", q / jnp.sqrt(q.shape[-1]), k[neighbours])
        dist = ((qp[:, None] - kp[neighbours]).to_array() ** 2).sum(axis=(-1, -2))
        bias = Linear(self.heads, bias=False, initializer="linear")(pair)
        attn_logits = w_L * (attn_logits - scale * dist + bias)
        attn_logits = jnp.where(mask[..., None], attn_logits, -1e9)
        attn = jax.nn.softmax(attn_logits, axis=1)
        attn = jnp.where(mask[..., None], attn, 0)

        # value
        local_update = jnp.einsum("ijh,ijhc->ihc", attn, v[neighbours]).reshape(attn.shape[0], -1)
        pair_update = jnp.einsum("ijh,ijc->ihc", attn, pair).reshape(attn.shape[0], -1)
        point_update = to_local(
            jnp.einsum("ijh,ijhcd->ihcd", attn, vp.to_array()[neighbours]))
        point_update = point_update.reshape(attn.shape[0], -1)
        result = jnp.concatenate((local_update, pair_update, point_update), axis=-1)
        return Linear(local.shape[-1], bias=False, initializer=self.final_init)(result)

class SparseAttention(hk.Module):
    """Sparse attention without point bias."""
    def __init__(self, size=32, heads=4,
                 final_init="zeros", normalize=False,
                 name: Optional[str]="sparse_attn"):
        super().__init__(name=name)
        self.size = size
        self.heads = heads
        self.final_init = final_init
        self.normalize = normalize

    def __call__(self, local, pair, neighbours, mask):
        if self.normalize:
            local = hk.LayerNorm([-1], True, True)(local)
        qkv = Linear(
            self.heads * 3 * self.size,
            bias=False, name="qkv"
        )(
            local
        ).reshape(list(local.shape[:-1]) + [self.heads, 3 * self.size])
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = hk.LayerNorm([-1], True, True)(q)
        k = hk.LayerNorm([-1], True, True)(k)

        bias = Linear(self.heads, bias=False, name="bias")(
            pair
        )
        if bias.ndim < local.ndim:
            extension = (local.ndim - bias.ndim) * [1]
            bias = bias.reshape(
                bias.shape[0], *extension, *bias.shape[1:]
            )
        w_L = jnp.sqrt(1 / 2)

        dot = jnp.sqrt(1 / self.size) * jnp.einsum("ihc,ijhc->ijh", q, k[neighbours])
        attn_logits = w_L * (dot + bias)
        if neighbours is None:
            pair_mask = mask
        else:
            pair_mask = mask * (neighbours != -1)
        attn_logits = jnp.where(pair_mask[..., None], attn_logits, -1e9)

        attn = jax.nn.softmax(attn_logits, axis=-2)
        attn = jnp.where(pair_mask[..., None], attn, 0.0)
        out_pair = jnp.einsum("ijh,ijc->ihc", attn, pair)
        out_scalar = jnp.einsum("ijh,ijhc->ihc", attn, v[neighbours])
        out = Linear(size=local.shape[-1], initializer=self.final_init, name="project_out")(
            jnp.concatenate((
                out_pair.reshape(*out_pair.shape[:-2], -1),
                out_scalar.reshape(*out_scalar.shape[:-2], -1)
            ), axis=-1)
        )

        return out.astype(local.dtype)

class DenseNonEquivariantPointAttention(hk.Module):
    """EXPERIMENTAL. Non-equivariant point attention"""
    def __init__(self, size=32, heads=4,
                 query_points=8, value_points=8,
                 final_init="zeros", normalize=False,
                 use_pair=False,
                 name: Optional[str]="ada_point_attention"):
        super().__init__(name=name)
        self.size = size
        self.heads = heads
        self.query_points = query_points
        self.value_points = value_points
        self.final_init = final_init
        self.use_pair = use_pair
        self.normalize = normalize

    def __call__(self, local, pos, resi, chain, batch, mask):
        if self.normalize:
            local = hk.LayerNorm([-1], True, True)(local)
        def attention_component(x, size=self.size, heads=self.heads):
            result = Linear(size * heads, initializer="linear")(x)
            return result.reshape(*result.shape[:-1], heads, self.size)
        def attention_point(x, pos, size=self.query_points, heads=self.heads):
            result = Linear(size * heads * 3, initializer="zeros")(x)
            result = result.reshape(*result.shape[:-1], heads, size, 3)
            result += pos[:, 1][:, None, None]
            return result
        def fourier_resi_embedding(x, size=128):
            val = x[..., None] / 10_000 ** (2 * jnp.arange(size // 2) / size)
            return jnp.concatenate((jnp.sin(val), jnp.cos(val)), axis=-1)
        def fourier_xyz_embedding(x, size=128):
            val = x[..., None] / 10_000 ** (2 * jnp.arange(size // 2) / size)
            return jnp.concatenate((jnp.sin(val), jnp.cos(val)), axis=-1).mean(axis=-2)
        same_batch = batch[:, None] == batch[None, :]
        pair_mask = mask[:, None] * mask[None, :] * same_batch

        query = hk.LayerNorm([-1], True, True)(attention_component(local))
        key = hk.LayerNorm([-1], True, True)(attention_component(local))
        value = attention_component(local)
        query_points = attention_point(local, pos)
        key_points = attention_point(local, pos)
        value_points = attention_point(local, pos)
        value = jnp.concatenate((value, value_points.reshape(*value.shape[:-1], -1)), axis=-1)
        inner_product_attention = jnp.einsum("ihc,jhc->ijh", query * jnp.sqrt(1 / query.shape[-1]), key)
        point_attention = -((query_points[:, None] - key_points[None, :]) ** 2).sum(axis=(-1, -2))
        resi_dist = jnp.clip(resi[:, None] - resi[None, :], -32, 32)# / 64
        other_chain = (chain[:, None] != chain[None, :])

        ca = Vec3Array.from_array(10 * pos[:, 1])
        dist = (ca[:, None] - ca[None, :]).norm()
        dist = distance_rbf(dist, bins=16)
        rel = (query_points[:, None] - key_points[None, :]).reshape(*dist.shape[:2], -1)
        rdist = resi_dist + 32
        rdist = jnp.where(other_chain, 65, rdist)
        rdist = hk.get_parameter("resi_embedding", (66, self.size + 3 * self.value_points,), init=init_linear())[rdist]
        pair = rdist + Linear(
            self.size + 3 * self.value_points,
            bias=False, initializer="linear")(jnp.concatenate((rel, dist), axis=-1))
        bias = Linear(self.heads)(jnp.concatenate((rel, dist), axis=-1))
        pair = pair[:, :, None, :] + value[None, :]
        
        w_C = jnp.sqrt(2 / (9 * self.query_points))

        point_scale = hk.get_parameter(
            "gamma", (self.heads,),
            init=hk.initializers.Constant(jnp.log(jnp.exp(1.) - 1.))
        )
        # single pair operation
        point_scale = jax.nn.softplus(point_scale.reshape(1, 1, self.heads)) * w_C / 2
        attn = (inner_product_attention + point_scale * point_attention + bias) * jnp.sqrt(1 / 3)
        attn = jnp.where(pair_mask[..., None], attn, -1e9)
        attn = jax.nn.softmax(attn, axis=1)
        attn = jnp.where(pair_mask[..., None], attn, 0)
        # add pair value
        result = jnp.einsum("ijh,ijhc->ihc", attn, pair).reshape(local.shape[0], -1)
        return Linear(local.shape[-1], bias=False, initializer="zeros")(result) 

class SparseSemiEquivariantPointAttention(hk.Module):
    """Sparse IPA with additional non-equivariant features."""
    def __init__(self, size=32, heads=4,
                 query_points=8, value_points=8,
                 final_init="zeros", normalize=False,
                 name: Optional[str]="ada_point_attention"):
        super().__init__(name=name)
        self.size = size
        self.heads = heads
        self.query_points = query_points
        self.value_points = value_points
        self.final_init = final_init
        self.normalize = normalize

    def __call__(self, local, pair, pos, neighbours, mask):
        # this module is only translation equivariant
        if self.normalize:
            local = hk.LayerNorm([-1], True, True)(local)
        qkv = Linear(
            self.heads * 3 * self.size,
            bias=False, name="qkv"
        )(
            local
        ).reshape(list(local.shape[:-1]) + [self.heads, 3 * self.size])
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = hk.LayerNorm([-1], True, True)(q)
        k = hk.LayerNorm([-1], True, True)(k)
        # global qkv as an offset from the base positions
        qkv_g = Linear(self.heads * (2 * self.query_points + self.value_points) * 3)(local)
        qkv_g = qkv.reshape(qkv.shape[0], -1, 3)
        qkv_g += pos[:, None, 1]
        qkv_g = qkv_g.reshape(*qkv_g.shape[:-2], self.heads, -1, 3)
        q_g, k_g, v_g = jnp.split(
            qkv_g,
            [self.query_points, 2 * self.query_points],
            axis=-2
        )
        bias = Linear(self.heads, bias=False, name="bias")(
            pair
        )
        if bias.ndim < local.ndim:
            extension = (local.ndim - bias.ndim) * [1]
            bias = bias.reshape(
                bias.shape[0], *extension, *bias.shape[1:]
            )
        w_C = jnp.sqrt(2 / (9 * self.query_points))
        w_L = jnp.sqrt(1 / 3)

        gamma = hk.get_parameter(
            "gamma", (self.heads,),
            init=hk.initializers.Constant(jnp.log(jnp.exp(1.) - 1.))
        )
        dfactor = jax.nn.softplus(gamma.reshape(1, 1, self.heads)) * w_C / 2
        dist = dfactor * jnp.square(q_g[:, None] - k_g[neighbours]).sum(axis=(-1, -2))
        dot = jnp.sqrt(1 / self.size) * jnp.einsum("ihc,ijhc->ijh", q, k[neighbours])
        attn_logits = w_L * (dot + bias - dist)
        if neighbours is None:
            pair_mask = mask
        else:
            pair_mask = mask * (neighbours != -1)
        attn_logits = jnp.where(pair_mask[..., None], attn_logits, -1e9)

        attn = jax.nn.softmax(attn_logits, axis=-2)
        attn = jnp.where(pair_mask[..., None], attn, 0.0)
        out_pair = jnp.einsum("ijh,ijc->ihc", attn, pair)
        out_scalar = jnp.einsum("ijh,ijhc->ihc", attn, v[neighbours])
        out_point = jnp.einsum("ijh,ijhpc->ihpc", attn, v_g[neighbours])
        out_point = Vec3Array.from_array(out_point.astype(jnp.float32))
        out_point: Vec3Array = out_point - Vec3Array.from_array(pos[:, 1])[:, None, None]
        out_norm = out_point.norm()
        out_point = out_point.to_array()
        out = Linear(size=local.shape[-1], initializer=self.final_init, name="project_out")(
            jnp.concatenate((
                out_pair.reshape(*out_pair.shape[:-2], -1),
                out_scalar.reshape(*out_scalar.shape[:-2], -1),
                out_point.reshape(*out_point.shape[:-3], -1),
                out_norm.reshape(*out_point.shape[:-3], -1)
            ), axis=-1)
        )

        return out.astype(local.dtype)

class SparseInvariantPointAttention(hk.Module):
    """Sparse IPA."""
    def __init__(self, size=32, heads=4,
                 query_points=8, value_points=8,
                 final_init="zeros", normalize=False,
                 name: Optional[str]="ada_point_attention"):
        super().__init__(name=name)
        self.size = size
        self.heads = heads
        self.query_points = query_points
        self.value_points = value_points
        self.final_init = final_init
        self.normalize = normalize

    def __call__(self, local, pair, frames, neighbours, mask):
        if self.normalize:
            local = hk.LayerNorm([-1], True, True)(local)
        frames: Rigid3Array = Rigid3Array.from_array(frames.astype(jnp.float32))
        qkv = Linear(
            self.heads * 3 * self.size,
            bias=False, name="qkv"
        )(
            local
        ).reshape(list(local.shape[:-1]) + [self.heads, 3 * self.size])
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = hk.LayerNorm([-1], True, True)(q)
        k = hk.LayerNorm([-1], True, True)(k)
        qkv_g = LinearToPoints(
            self.heads * (2 * self.query_points + self.value_points),
            name="qkv_global"
        )(local, frames)
        qkv_g = qkv_g.reshape(*qkv_g.shape[:-2], self.heads, -1, 3)
        q_g, k_g, v_g = jnp.split(
            qkv_g,
            [self.query_points, 2 * self.query_points],
            axis=-2
        )
        bias = Linear(self.heads, bias=False, name="bias")(
            pair
        )
        if bias.ndim < local.ndim:
            extension = (local.ndim - bias.ndim) * [1]
            bias = bias.reshape(
                bias.shape[0], *extension, *bias.shape[1:]
            )
        w_C = jnp.sqrt(2 / (9 * self.query_points))
        w_L = jnp.sqrt(1 / 3)

        gamma = hk.get_parameter(
            "gamma", (self.heads,),
            init=hk.initializers.Constant(jnp.log(jnp.exp(1.) - 1.))
        )
        dfactor = jax.nn.softplus(gamma.reshape(1, 1, self.heads)) * w_C / 2
        dist = dfactor * jnp.square(q_g[:, None] - k_g[neighbours]).sum(axis=(-1, -2))
        dot = jnp.sqrt(1 / self.size) * jnp.einsum("ihc,ijhc->ijh", q, k[neighbours])
        attn_logits = w_L * (dot + bias - dist)
        if neighbours is None:
            pair_mask = mask
        else:
            pair_mask = mask * (neighbours != -1)
        attn_logits = jnp.where(pair_mask[..., None], attn_logits, -1e9)

        attn = jax.nn.softmax(attn_logits, axis=-2)
        attn = jnp.where(pair_mask[..., None], attn, 0.0)
        out_pair = jnp.einsum("ijh,ijc->ihc", attn, pair)
        out_scalar = jnp.einsum("ijh,ijhc->ihc", attn, v[neighbours])
        out_point = jnp.einsum("ijh,ijhpc->ihpc", attn, v_g[neighbours])
        out_point = Vec3Array.from_array(out_point.astype(jnp.float32))
        out_point: Vec3Array = frames[:, None, None].apply_inverse_to_point(out_point)
        out_norm = out_point.norm()
        out_point = out_point.to_array()
        out = Linear(size=local.shape[-1], initializer=self.final_init, name="project_out")(
            jnp.concatenate((
                out_pair.reshape(*out_pair.shape[:-2], -1),
                out_scalar.reshape(*out_scalar.shape[:-2], -1),
                out_point.reshape(*out_point.shape[:-3], -1),
                out_norm.reshape(*out_point.shape[:-3], -1)
            ), axis=-1)
        )

        return out.astype(local.dtype)

def atom_pool(factor=8, num_in=64, num_out=8):
    """DEPRECATED."""
    def inner(positions, resi, chain, batch: jnp.ndarray, mask: jnp.ndarray):
        positions = Vec3Array.from_array(positions)
        # random subset of residues
        centers = jax.random.permutation(
            hk.next_rng_key(), positions.shape[0])[:positions.shape[0] // factor]
        centers = jnp.sort(centers, axis=0)
        same_batch = batch[centers, :] == batch[None, :]
        pair_mask = mask[centers, :] * mask[None, :] * same_batch
        distance = (positions[centers][:, None] - positions[None, :]).norm()
        distance = jnp.where(pair_mask, distance, jnp.inf)
        in_neighbours = get_neighbours(num_in)(distance, mask, None)
        out_neighbours = get_neighbours(num_out)(distance.T, pair_mask.T, None)
        return centers, in_neighbours, out_neighbours
    return inner

def sum_equivariant_pair_embedding(config, use_local=True):
    c = config
    def inner(local, pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours] * (neighbours != -1)
        frames, _ = extract_aa_frames(pos)
        features = 0.0
        if c.pair_vector_features:
            features += Linear(c.pair_size, bias=False)(
                frame_pair_vector_features(
                    frames, pos, neighbours,
                    c.position_scale))
        features += Linear(c.pair_size, bias=False)(
            sequence_relative_position(
                c.relative_position_encoding_max, one_hot=True,
                cyclic=c.cyclic, identify_ends=c.identify_ends)(
                    resi, chain, batch, neighbours=neighbours
                ))
        if use_local:
            local_features = Linear(
                c.pair_size, bias=False)(local)[:, None]
            local_features += Linear(
                c.pair_size, bias=False)(local)[neighbours]
            features += local_features
        features = hk.LayerNorm([-1], True, True)(features)
        features = jnp.where(pair_mask[..., None], features, 0)
        return features, pair_mask
    return inner

def equivariant_pair_embedding(config, use_local=True):
    c = config
    def inner(local, pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours] * (neighbours != -1)
        frames, _ = extract_aa_frames(pos)
        features = [frame_pair_features(frames, pos, neighbours)]
        if c.pair_vector_features:
            features.append(frame_pair_vector_features(
                    frames, pos, neighbours,
                    c.position_scale))
        features.append(sequence_relative_position(
            c.relative_position_encoding_max, one_hot=True,
            cyclic=c.cyclic, identify_ends=c.identify_ends)(
                resi, chain, batch, neighbours=neighbours
            ))
        if use_local:
            local_features = Linear(
                c.pair_size, bias=True,
                initializer=init_glorot())(local)[:, None]
            local_features += Linear(
                c.pair_size, bias=True,
                initializer=init_glorot())(local)[neighbours]
            features.append(local_features)
        features = jnp.concatenate(features, axis=-1)
        features = Linear(c.pair_size, initializer=init_glorot())(features)
        features = hk.LayerNorm([-1], True, True)(features)
        features = jnp.where(pair_mask[..., None], features, 0)
        return features, pair_mask
    return inner
