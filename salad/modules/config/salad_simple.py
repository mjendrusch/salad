from salad.modules.utils.collections import dotdict, deepcopy

small = dotdict(
    ## feature sizes
    local_size=784,
    ## module parameters
    factor=4,
    heads=32,
    key_size=32,
    use_attn=True,
    ## diffusion stack parameters
    diffusion_depth=8,
    block_size=1,
    # position diffusion parameters
    diffusion_kind="flow",
    pos_mean_sigma=1.6,
    pos_std_sigma=1.4,
    sigma_data=10.0,
    preconditioning="flow",
    # loss weights
    pos_weight=2.0,
    trajectory_weight=0.0,
    aa_weight=10.0,
    # dataset constraints
    min_size=50,
    max_size=None
)

tiny = deepcopy(small)
tiny.diffusion_depth = 1
tiny.add_pos_embedding = True

tiny_no_attn = deepcopy(tiny)
tiny_no_attn.use_attn = False

medium = deepcopy(small)
medium.local_size = 1024
medium.diffusion_depth = 18

small_pred_pos = deepcopy(small)
small_pred_pos.embed_pred_pos = True
small_pred_pos.trajectory_weight = 1.0

small_tim = deepcopy(small)
small_tim.tim_target = True