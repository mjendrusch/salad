from copy import deepcopy
from salad.modules.utils.collections import dotdict

default = dotdict(
    local_size=128,
    pair_size=128,
    factor=2,
    depth=6,
    eval=False,
    distogram_min=2.0,
    distogram_max=22.0,
    max_resi_distance=32,
    distogram_bins=64,
    noise_level=0.3,
    drop_rate=0.1,
    layer_norm="prenorm",
)

large = deepcopy(default)
large.update(
    pair_size=256,
    depth=8
)

large_noisy = deepcopy(large)
large_noisy.update(
    noise_level=0.5,
    drop_rate=0.25,
)

large_postnorm = deepcopy(large)
large_postnorm.update(
    noise_level=0.5,
    drop_rate=0.25,
    layer_norm="postnorm"
)

xlarge = deepcopy(default)
xlarge.update(
    pair_size=256,
    depth=10
)

tiny = deepcopy(default)
tiny.update(
    pair_size=64,
    depth=3,
)

test = deepcopy(tiny)
test.update(
    local_size=16,
    pair_size=16,
    depth=1,
)