from salad.modules.utils.collections import dotdict, deepcopy
import numpy as np

default = dotdict(
    ## feature sizes
    local_size=256,
    pair_size=64,
    latent_size=20,
    noembed=True,
    ## module parameters
    relative_position_encoding_max=32,
    factor=4,
    heads=8,
    key_size=32,
    multi_query=False,
    sigma_data=10.0,
    ## diffusion stack parameters
    depth=6,
    block_size=1,
    num_recycle=3,
    ## aa diffusion parameters
    # no sequence diffusion
    aa_decoder_depth=2,
    encoder_depth=4,
    ## loss parameters
    # loss clipping
    p_clip=1.0,
    clip_fape=100,
    no_fape2=True,
    # time embedding
    time_embedding=True,
    # local loss parameters
    local_neighbours=16,
    fape_neighbours=64,
    # loss weights
    local_weight=1.0,
    aa_weight=10.0,
    fape_weight=10.0,
    fape_trajectory_weight=10 * 0.5,
    violation_scale=0.1,
    # dataset constraints
    min_size=50,
    max_size=512 # FIXME
)

noresi = deepcopy(default)
noresi.no_resi_encoder = True

quant = deepcopy(noresi)
quant.quantize = True
