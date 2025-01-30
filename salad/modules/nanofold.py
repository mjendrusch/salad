import jax
import jax.numpy as jnp
import haiku as hk

from modules.basic import MLP, Linear, init_zeros, init_relu, init_linear

class NanoFold(hk.Module):
    def __init__(self, config, name: str | None = "nanofold"):
        super().__init__(name)

    def __call__(self, data):
        c = self.config
        encoder = SequenceEncoder(c) # sequence encoder
        diffusion = StructureDiffusion(c) # diffusion model
        confidence = ConfidenceHead(c) # confidence head
        sup_neighbours = ... # TODO

        # encoder recycling
        def encoder_iteration(i, prev):
            local, *_ = encoder(data, prev)
            return local
        count = jax.random.randint(hk.next_rng_key(), (), 0, 4)
        init_prev = jnp.zeros((data["aa"].shape[0], c.local_size), dtype=jnp.float32)
        prev = hk.fori_loop(0, count, encoder_iteration, init_prev)
        prev = jax.lax.stop_gradient(prev)
        local, aa_prediction, pair_aa_prediction, distogram_trajectory = encoder(
            data, prev, sup_neighbours=sup_neighbours)

        # diffusion rollout
        rollout_data = diffusion.init_diffusion(data, local)
        def diffusion_rollout_step(data):
            pos_0 = diffusion(data, local)

class SequenceEncoder(hk.Module):
    def __init__(self, config, name: str | None = "seq_encoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data, prev, sup_neighbours=None):
        c = self.config
        encoder_stack = EncoderStack(c)
        local, resi, chain, batch, mask = self.prepare_features(data, prev)
        local, distograms = encoder_stack(local, resi, chain, batch, mask,
                                          sup_neighbours=sup_neighbours)
        local = hk.LayerNorm([-1], False, False)(local)
        aa_prediction = MLP(local.shape[-1] * 2, 20, bias=False,
                            activation=jax.nn.gelu, final_init=init_zeros())(local)
        pair_aa_prediction = InnerAA(c)(local, data["aa_gt"])
        return local, aa_prediction, pair_aa_prediction, distograms

class InnerAA(hk.Module):
    def __init__(self, config, name: str | None = "inner_distogram"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, aa_gt):
        features = MLP(local.shape[-1] * 2, local.shape,
                       bias=False, activation=jax.nn.gelu,
                       final_init=init_linear())(jnp.concatenate((
                           local, jax.nn.one_hot(aa_gt, 20, axis=-1)), axis=-1))
        dcode = Linear(20 * 8, bias=False)(features).reshape(features.shape[0], 8, 20)
        dgate = Linear(20 * 8, bias=False)(features).reshape(features.shape[0], 8, 20)
        dgate = jax.nn.gelu(dgate)
        weight = hk.get_parameter("inner_weight", (8, 8, 20), init=hk.initializers.Constant(0.0))
        logits = jnp.einsum("iax,jbx,abx->ijx", dcode, dgate, weight)
        logits = jax.nn.log_softmax(logits, axis=-1)
        return logits