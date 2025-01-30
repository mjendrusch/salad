from salad.modules.utils.collections import dotdict, deepcopy

default = dotdict(
    local_size=256,
    pair_size=64,
    depth=12,
    block_size=4,
    noise_level=0.3,
    num_index=16,
    num_spatial=16,
    num_random=32,
    heads=8,
    key_size=32,
    factor=4,
    global_update=True
)

unweighted = deepcopy(default)
unweighted.unweighted = True
unweighted.depth = 6
unweighted.block_size = 1
unweighted.local_size = 128

tiny = deepcopy(unweighted)
tiny.global_update = False
tiny.num_index = 0
tiny.num_random = 0
tiny.num_spatial = 32
tiny.factor = 2
tiny.noise_once = True
