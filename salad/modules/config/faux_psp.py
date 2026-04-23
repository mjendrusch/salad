from copy import deepcopy
from salad.modules.utils.collections import dotdict

unordered = dotdict(
    encoder=dotdict(
        kind="UnorderedEncoder",
        local_size=128,
        pair_size=64,
        latent_size=32,
        factor=2,
        depth=2,
        heads=8,
        key_size=32,
        num_neighbours=48,
        noise_level=0.02
    ),
    decoder=dotdict(
        kind="Decoder",
        local_size=128,
        pair_size=64,
        heads=8,
        key_size=32,
        pair_update=True,
        factor=2,
        depth=3,
        losses=dotdict(
            distogram=1.0,
            distogram_short=None,
            contact=10.0,
            contact_short=None,
            dssp=1.0,
            dssp_short=None,
            aa=1.0,
            aa_short=1.0,
            aa_pair=1.0,
            aa_pair_short=None,
            ramagram=1.0,
            orientogram=1.0,
            plddt=0.1,
            pae=0.01
        )
    ),
    diagnostics=dotdict(
        crop_size=256,
        pair_size=32,
        relative_order=1.0,
        absolute_order=1.0,
        distogram=1.0,
        contact=1.0,
    )
)

unordered_ogram = deepcopy(unordered)
unordered_ogram.decoder.orientogram = True
unordered_ogram.decoder.ramagram = True
unordered_ogram.decoder.symmetrize = True

unordered_lnorm = deepcopy(unordered_ogram)
unordered_lnorm.encoder.normalize_input = True
unordered_lnorm.decoder.normalize_input = True

unordered_struc = deepcopy(unordered_lnorm)
unordered_struc.decoder.structure_module = True
unordered_struc.decoder.losses.fape = 10.0

unordered_struc_error = deepcopy(unordered_struc)
unordered_struc_error.decoder.error = True
unordered_struc_error.decoder.error_depth = 1
unordered_struc_error.decoder.recycle = True

unordered_struc_error_ind = deepcopy(unordered_struc_error)
unordered_struc_error_ind.drop_latent = True
unordered_struc_error_ind.decoder.structure_independent = True
unordered_struc_error_ind.decoder.error_independent = True

unordered_struc_error_large = deepcopy(unordered_struc_error_ind)
unordered_struc_error_large.decoder.structure_independent = False
unordered_struc_error_large.decoder.error_independent = False
unordered_struc_error_large.decoder.depth = 12

unordered_struc_error_large_ft = deepcopy(unordered_struc_error_large)
unordered_struc_error_large_ft.decoder.losses.violation = 0.1

unordered_struc_error_large_enoise = deepcopy(unordered_struc_error_large)
unordered_struc_error_large_enoise.encoder.encoding_noise = True
unordered_struc_error_large_enoise.encoder.latent_size = 8
unordered_struc_error_large_enoise.drop_latent = False

unordered_struc_error_large_noisy = deepcopy(unordered_struc_error_large)
unordered_struc_error_large_noisy.noise_mean = 0.5
unordered_struc_error_large_noisy.noise_std = 1.2
unordered_struc_error_large_noisy.denoise = True

aa_struc_error_ind = deepcopy(unordered_struc_error_ind)
aa_struc_error_ind.drop_latent = False
aa_struc_error_ind.encoder.kind = "AAEncoder"
aa_struc_error_ind.decoder.structure_independent = False

aa_struc_error_large = deepcopy(aa_struc_error_ind)
aa_struc_error_large.decoder.depth = 12
aa_struc_error_large.decoder.structure_independent = False
aa_struc_error_large.decoder.error_independent = False

aa_struc_error_large_hard = deepcopy(aa_struc_error_large)
aa_struc_error_large_hard.encoder.noise_level = 0.3
aa_struc_error_large_hard.temperature = 0.5
aa_struc_error_large_hard.sample = True

unordered_struc_error_mix = deepcopy(unordered_struc_error_ind)
unordered_struc_error_mix.mix_latent = True
unordered_struc_error_mix.decoder.losses["violation"] = 0.1

unordered_struc_error_mix_large = deepcopy(unordered_struc_error_mix)
unordered_struc_error_mix_large.decoder.error_depth = 4
unordered_struc_error_mix_large.decoder.losses["violation"] = 0.1

unordered_large = deepcopy(unordered)
unordered_large.decoder.depth = 8
