import numpy as np

import jax
import jax.numpy as jnp

class Screw:
    def __init__(self, count=3, angle=0, radius=15.0, translation=0.0, chain_center=False):
        angle = angle / 180 * np.pi
        self.count = count
        self.angle = angle
        self.radius = radius
        self.translation = translation
        self.chain_center = chain_center
        # prepare rotation matrices
        self.rot = jnp.array([
            [np.cos(angle),  0.0000000, -np.sin(angle)],
            [    0.0000000,  1.0000000,      0.0000000],
            [np.sin(angle),  0.0000000,  np.cos(angle)]])
        self.rot_x = [jnp.eye(3), self.rot]
        for _ in range(count - 2):
            self.rot_x.append(jnp.einsum("ac,cb->ab", self.rot_x[-1], self.rot))

    def replicate_pos(self, first: jnp.ndarray, do_radius: bool):
        # replicate positions
        first = first.at[:].add(jnp.array([do_radius * self.radius, 0.0, 0.0]))
        replicates = []
        for idx in range(self.count):
            replicates.append(
                jnp.einsum("...c,cd->...d", first, self.rot_x[idx])
            + idx * jnp.array([0.0, self.translation, 0.0]))
        result = jnp.concatenate(replicates, axis=0)
        return result - result[:, 1].mean(axis=0)

    def couple_pos(self, x: jnp.ndarray, do_radius: bool, center=None):
        # couple positions
        size = x.shape[0] // self.count
        representative = 0
        for idx in range(self.count):
            representative += jnp.einsum(
                "...c,cd->...d",
                cyclic_centering(
                    x[idx * size:(idx + 1) * size],
                    docenter=do_radius, center=center),
                self.rot_x[idx].T)
        representative /= self.count
        return representative

    def select_pos(self, x: jnp.ndarray, do_radius: bool, center=None):
        size = x.shape[0] // self.count
        idx = self.count // 2
        representative = jnp.einsum(
            "...c,cd->...d",
            cyclic_centering(
                x[idx * size:(idx + 1) * size],
                docenter=do_radius, center=center),
            self.rot_x[idx].T)
        return representative

    def replicate_features(self, data: jnp.ndarray):
        return jnp.concatenate(self.count * [data], axis=0)

    def couple_features(self, data: jnp.ndarray):
        size = data.shape[0]
        subsize = size // self.count
        units = data.reshape(self.count, subsize, *data.shape[1:])
        unit = units.mean(axis=0)
        return unit

    def select_features(self, data):
        size = data.shape[0]
        subsize = size // self.count
        idx = self.count // 2
        unit = data[idx * subsize:(idx + 1) * subsize]
        return unit

def cyclic_centering(x, docenter=True, center=None):
    if center is None:
        center = x[:, 1].mean(axis=0)
    return jnp.where(docenter, x.at[:].add(-center), x)

def rot_x(angle):
    return jnp.array([
        [1.0000000,      0.0000000,      0.0000000],
        [0.0000000,  np.cos(angle), -np.sin(angle)],
        [0.0000000,  np.sin(angle),  np.cos(angle)]])
def rot_y(angle):
    return jnp.array([
        [np.cos(angle),  0.0000000, -np.sin(angle)],
        [    0.0000000,  1.0000000,      0.0000000],
        [np.sin(angle),  0.0000000,  np.cos(angle)]])
def rot_z(angle):
    return jnp.array([
        [np.cos(angle), -np.sin(angle), 0.0000000],
        [np.sin(angle),  np.cos(angle), 0.0000000],
        [0.0000000,          0.0000000, 1.0000000]])
def apply_rot(rot, x):
    return jnp.einsum("...c,cd->...d", x, rot)
