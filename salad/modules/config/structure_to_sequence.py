from salad.modules.utils.collections import dotdict, deepcopy

default = dotdict(
    local_size=128,
    pair_size=64,
    depth=6,
    block_size=1,
    noise_level=0.3,
    num_neighbours=32,
    heads=8,
    key_size=32,
    factor=4,
    resi_dual=False,
    sigma_data=10.0,
    drop=False,
    recycle=True
)

default_single = deepcopy(default)
default_single.single_step = True

interface_3 = deepcopy(default)
interface_3.local_size = 256
interface_3.num_neighbours = 48
interface_3.drop = True
interface_3.norm_out = True

interface_6 = deepcopy(default)
interface_6.num_neighbours = 48
interface_6.resi_dual = True
interface_6.drop = True
interface_6.norm_out = True

interface_6p = deepcopy(default)
interface_6p.num_neighbours = 48
interface_6p.drop = True
interface_6p.persistent_pair = True
interface_6p.norm_out = True

interface_6pr = deepcopy(default)
interface_6pr.num_neighbours = 48
interface_6pr.drop = True
interface_6pr.resi_dual = True
interface_6pr.persistent_pair = True
interface_6pr.norm_out = True

interface_3pr = deepcopy(default)
interface_3pr.num_neighbours = 48
interface_3pr.depth = 3
interface_3pr.drop = True
interface_3pr.resi_dual = True
interface_3pr.persistent_pair = True
interface_3pr.norm_out = True

decoder_only_6 = deepcopy(default)
decoder_only_6.depth = 6
decoder_only_6.recycle = False
decoder_only_6.drop = True
decoder_only_6.predict_none = True

decoder_only_9 = deepcopy(decoder_only_6)
decoder_only_9.depth = 9

decoder_only_12 = deepcopy(decoder_only_6)
decoder_only_12.depth = 12

default_decoder = deepcopy(default)
default_decoder.decoder_depth = 3
default_decoder.drop = True
default_decoder.norm_out = True

default_decoder2 = deepcopy(default)
default_decoder2.decoder_depth = 3
default_decoder2.drop = True
default_decoder2.norm_out = False

default_decoder_6 = deepcopy(default_decoder2)
default_decoder_6.depth = 3

default_decoder_12 = deepcopy(default_decoder2)
default_decoder_12.depth = 9

medium = deepcopy(default)
medium.local_size = 128
medium.pair_size = 64
medium.depth = 12
