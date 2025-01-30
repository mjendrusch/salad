from salad.modules.utils.collections import dotdict, deepcopy

default = dotdict(
    local_size=256,
    pair_size=64,
    depth=8,
    block_size=1,
    noise_level=0.3,
    num_neighbours=32,
    heads=8,
    key_size=32,
    factor=4,
    resi_dual=True,
    sigma_data=10.0,
    drop=False,
    recycle=True
)

default_allcond = deepcopy(default)
default_allcond.p_condition = 1.0
default_allcond.drop = True

large = deepcopy(default)
large.p_condition = 1.0
large.drop = False
large.depth = 24
