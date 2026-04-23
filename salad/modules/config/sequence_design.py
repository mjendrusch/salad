from salad.modules.utils.collections import dotdict, deepcopy

default = dotdict(
    local_size=128,
    pair_size=128,
    depth=6,
    noise_scale=0.3,
    num_aa_neighbours=48,
    num_smol_neighbours=16,
    label_smoothing=0.0
)
