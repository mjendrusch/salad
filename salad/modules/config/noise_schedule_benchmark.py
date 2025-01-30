from salad.modules.utils.collections import dotdict, deepcopy

default_vp = dotdict(
    ## encoder parameters
    # 15 from augmented positions
    augment_size=15,
    encoder_depth=2,
    ## feature sizes
    local_size=128,
    pair_size=64,
    ## neighbours (64)
    num_neighbours=48,
    ## module parameters
    resi_dual=False,
    relative_position_encoding_max=32,
    factor=4,
    heads=8,
    key_size=32,
    multi_query=False, # True # use multi-query attention
    ## diffusion stack parameters
    diffusion_depth=6,
    block_size=1,
    # position diffusion parameters
    diffusion_kind="vpfixed",
    pos_mean_sigma=1.6,
    pos_std_sigma=1.4,
    sigma_data=10.0,
    preconditioning="vp",
    diffusion_time_scale="cosine",
    ## aa diffusion parameters
    # no sequence diffusion
    diffuse_sequence=False,
    aa_decoder_kind="adm",
    aa_decoder_depth=3,
    ## loss parameters
    # loss clipping
    p_clip=1.0,
    # dssp dropping
    p_drop_dssp=0.2,
    # time embedding
    time_embedding=True,
    # local loss parameters
    local_neighbours=16,
    fape_neighbours=64,
    # loss weights
    pos_weight=2.0,
    trajectory_weight=1.0,
    x_weight=2.0,
    rotation_weight=2.0,
    rotation_trajectory_weight=1.0,
    violation_weight=0.1,
    local_weight=10.0,
    aa_weight=10.0,
    fape_weight=1,
    fape_trajectory_weight=0.5,
    # dataset constraints
    min_size=50,
    max_size=None
)

multimotif_vp = deepcopy(default_vp)
multimotif_vp.multi_motif = True

nano_vp = deepcopy(default_vp)
nano_vp.multi_motif = True
nano_vp.repeat = True

binder_vp = deepcopy(default_vp)
binder_vp.encoder_depth = 3

binder_centered_vp = deepcopy(binder_vp)
binder_centered_vp.binder_centered = True

default_vp_omap = deepcopy(default_vp)
default_vp_omap.use_omap = True

default_vp_distogram = deepcopy(default_vp)
default_vp_distogram.distogram_trajectory = True

default_vp_minimal = deepcopy(default_vp)
default_vp_minimal.minimal = True

default_vp_minimal_timeless = deepcopy(default_vp_minimal)
default_vp_minimal_timeless.time_embedding = False

default_vp_minimal_timeless_omap = deepcopy(default_vp_minimal_timeless)
default_vp_minimal_timeless_omap.use_omap = True

default_vp_minimal_timeless_atom14 = deepcopy(default_vp_minimal_timeless)
default_vp_minimal_timeless_atom14.encode_atom14 = True
default_vp_minimal_timeless_atom14.augment_size = 0
default_vp_minimal_timeless_atom14.soft_lddt = True

default_vp_minimal_timeless_atom14a = deepcopy(default_vp_minimal_timeless)
default_vp_minimal_timeless_atom14a.encode_atom14 = True
default_vp_minimal_timeless_atom14a.augment_size = 0
default_vp_minimal_timeless_atom14a.soft_lddt = True
default_vp_minimal_timeless_atom14a.atom = True

default_vp_atom14_kabsch = deepcopy(default_vp_minimal_timeless_atom14a)
default_vp_atom14_kabsch.atom_size = 32
default_vp_atom14_kabsch.kabsch = True

default_vp_timeless = deepcopy(default_vp)
default_vp_timeless.time_embedding = False

default_vp_timeless_omap = deepcopy(default_vp_timeless)
default_vp_timeless_omap.use_omap = True

default_ve = deepcopy(default_vp)
default_ve.preconditioning = "edm"
default_ve.diffusion_kind = "edm"

default_ve_scaled = deepcopy(default_vp)
default_ve_scaled.preconditioning = "edm_scaled"
default_ve_scaled.diffusion_kind = "edm"
default_ve_scaled.pos_weight *= 100
default_ve_scaled.trajectory_weight *= 100

default_ve_scaled_omap = deepcopy(default_ve_scaled)
default_ve_scaled_omap.use_omap = True

default_ve_scaled_minimal = deepcopy(default_ve_scaled)
default_ve_scaled_minimal.minimal = True

default_ve_scaled_minimal_loss = deepcopy(default_ve_scaled)
default_ve_scaled_minimal_loss.minimal = True
default_ve_scaled_minimal_loss.pos_weight *= 100
default_ve_scaled_minimal_loss.trajectory_weight *= 100

default_ve_timeless = deepcopy(default_ve_scaled)
default_ve_timeless.time_embedding = False
default_ve_timeless.pos_weight *= 100
default_ve_timeless.trajectory_weight *= 100

default_ve_timeless_omap = deepcopy(default_ve_timeless)
default_ve_timeless_omap.use_omap = True

default_ve_minimal_timeless = deepcopy(default_ve_scaled_minimal_loss)
default_ve_minimal_timeless.time_embedding = False

default_ve_minimal_timeless_omap = deepcopy(default_ve_minimal_timeless)
default_ve_minimal_timeless_omap.use_omap = True

default_ve_minimal_timeless_atom14 = deepcopy(default_ve_minimal_timeless)
default_ve_minimal_timeless_atom14.encode_atom14 = True
default_ve_minimal_timeless_atom14.augment_size = 0
default_ve_minimal_timeless_atom14.soft_lddt = True

default_ve_atom14_ca = deepcopy(default_ve_minimal_timeless_atom14)
default_ve_atom14_ca.atom14_ca = True
default_ve_atom14_ca.mask_atom14 = True
default_ve_atom14_ca.aa_trajectory = True

default_ve_atom14_canomask = deepcopy(default_ve_minimal_timeless_atom14)
default_ve_atom14_canomask.atom14_ca = True
default_ve_atom14_canomask.aa_trajectory = True

default_ve_atom14_learned = deepcopy(default_ve_minimal_timeless_atom14)
default_ve_atom14_learned.atom14_learned = True
default_ve_atom14_learned.aa_trajectory = True

default_ve_atom14 = deepcopy(default_ve_minimal_timeless)
default_ve_atom14.time_embedding = True
default_ve_atom14.encode_atom14 = True
default_ve_atom14.augment_size = 0
default_ve_atom14.soft_lddt = True
default_ve_atom14.aa_trajectory = True

default_ve_atom14_mask = deepcopy(default_ve_atom14)
default_ve_atom14_mask.mask_atom14 = True

large_vp = deepcopy(default_vp)
large_vp.diffusion_depth = 8
large_vp.local_size = 384
large_vp.pair_size = 128

default_vp_scaled = deepcopy(default_vp)
default_vp_scaled.diffusion_kind = "vp"

default_vp_scaled_omap = deepcopy(default_vp_scaled)
default_vp_scaled_omap.use_omap = True

default_vp_scaled_timeless = deepcopy(default_vp_scaled)
default_vp_scaled_timeless.time_embedding = False

default_vp_scaled_timeless_omap = deepcopy(default_vp_scaled_timeless)
default_vp_scaled_timeless_omap.use_omap = True

default_vp_scaled_minimal_timeless = deepcopy(default_vp_scaled_timeless)
default_vp_scaled_minimal_timeless.minimal = True

default_vp_scaled_minimal_timeless_omap = deepcopy(default_vp_scaled_minimal_timeless)
default_vp_scaled_minimal_timeless_omap.use_omap = True

default_vp_scaled_minimal = deepcopy(default_vp_scaled)
default_vp_scaled_minimal.minimal = True

default_flow = deepcopy(default_vp)
default_flow.diffusion_kind = "flow"
default_flow.preconditioning = "flow"
default_flow.diffusion_time_scale = "none"

default_flow_scaled = deepcopy(default_vp)
default_flow_scaled.diffusion_kind = "flow_scaled"
default_flow_scaled.preconditioning = "flow"
default_flow_scaled.diffusion_time_scale = "none"

semi_equivariant_vp = deepcopy(default_vp)
semi_equivariant_vp.equivariance = "semi_equivariant"

semi_equivariant_kabsch_vp = deepcopy(semi_equivariant_vp)
semi_equivariant_kabsch_vp.kabsch = True

noaugment_vp = deepcopy(default_vp)
noaugment_vp.augment_size = 0

linear_vp = deepcopy(default_vp)
linear_vp.augment_size = 0
linear_vp.linear_aa = True
