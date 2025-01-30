from salad.modules.utils.collections import dotdict, deepcopy

edm_backbone_augmented_refine = dotdict(
    ## encoder parameters
    # total of 20 positions
    # 5 from NCaCOCb backbone
    input_structure_kind="backbone",
    # 15 from augmented positions
    augment_size=15,
    embed_residues=False,
    encoder_depth=4,
    ## feature sizes
    local_size=256,
    pair_size=64,
    ## neighbours (64)
    recompute_neighbours=True,
    index_neighbours=16,
    spatial_neighbours=16,
    random_neighbours=32,
    ## module parameters
    # use ResiDual parameterisation
    resi_dual=True,
    use_attention=True,
    global_update=True,
    pair_vector_features=True,
    # identify -32 and 32 distances + other-chain
    # in relative position encoding
    identify_ends=False,
    relative_position_encoding_max=32,
    factor=4,
    heads=8,
    key_size=32,
    use_initial_position=True,
    # update positions in each block
    refine=True,
    position_scale=10.0,
    ## diffusion stack parameters
    diffusion_depth=6,
    block_size=1,
    # position diffusion parameters
    diffusion_kind="edm",
    block_type="edm",
    scale_to_chain_center=False,
    pos_mean_sigma=1.6,
    pos_std_sigma=1.4,
    seq_min_sigma=0.01,
    seq_max_sigma=50.0,
    sigma_data=10.0,
    preconditioning="edm",
    diffusion_time_scale="cosine",
    ## aa diffusion parameters
    # no sequence diffusion
    diffuse_sequence=False,
    sequence_noise_scale=4.0,
    seq_factor=1.0,
    seq_size=32,
    aa_decoder_kind="linear",
    aa_decoder_depth=2,
    ## condition parameters
    pair_condition=True,
    ## output parameters
    # output sidechain positions defined by torsion angles
    output_structure_kind="angle",
    ## loss parameters
    trajectory_discount=1.0,
    # loss clipping
    p_clip=0.7,
    # dssp dropping
    p_drop_dssp=0.2,
    # time embedding
    time_embedding=True,
    # loss weights
    pos_weight=200.0,
    x_weight=200.0,
    rotation_weight=1.0,
    trajectory_weight=100.0,
    violation_weight=0.1,
    local_weight=10.0,
    aa_weight=10.0,
    fape_weight=0.1,
    seq_weight=0,
    decoder_weight=0,
    # dataset constraints
    min_size=32,
    max_size=None
)

edm_backbone_augmented_simple = deepcopy(edm_backbone_augmented_refine)
edm_backbone_augmented_simple.encoder_depth = 2
edm_backbone_augmented_simple.pair_vector_features = False
edm_backbone_augmented_simple.augment_size = 32
edm_backbone_augmented_simple.block_size = 1
edm_backbone_augmented_simple.local_weight = 0
edm_backbone_augmented_simple.pair_condition = False
edm_backbone_augmented_simple.sum_features = False
edm_backbone_augmented_simple.vector_norm = "mean_norm"

edm_backbone_preaugment = deepcopy(edm_backbone_augmented_simple)
edm_backbone_preaugment.pre_augment = True
edm_backbone_preaugment.fape_weight = 0
edm_backbone_preaugment.local_weight = 0
edm_backbone_preaugment.encoder_depth = 4

edm_backbone_preaugment_noresi = deepcopy(edm_backbone_preaugment)
edm_backbone_preaugment_noresi.no_relative_position = True
edm_backbone_preaugment_noresi.encoder_depth = 2
edm_backbone_preaugment_noresi.index_neighbours = 0
edm_backbone_preaugment_noresi.spatial_neighbours = 32

# Framediff-scaled EDM.
# The noise distribution is heavily skewed towards larger noise
# This will hopefully force the model to do more work at high
# noise levels
fedm_backbone_preaugment = deepcopy(edm_backbone_preaugment)
fedm_backbone_preaugment.diffusion_kind = "fedm"
fedm_backbone_preaugment.encoder_depth = 2

edm_backbone_preaugment_pair = deepcopy(edm_backbone_preaugment)
edm_backbone_preaugment_pair.encoder_depth = 2
edm_backbone_preaugment_pair.fape_weight = 0
edm_backbone_preaugment_pair.pair_condition = True

edm_backbone_preaugment_pair_norefine = deepcopy(edm_backbone_preaugment_pair)
edm_backbone_preaugment_pair_norefine.refine = False

edm_backbone_preaugment_pair_notrajectoryloss = deepcopy(edm_backbone_preaugment_pair)
edm_backbone_preaugment_pair_notrajectoryloss.no_trajectory_loss = True

vp_backbone_preaugment = deepcopy(edm_backbone_augmented_simple)
vp_backbone_preaugment.pre_augment = True
vp_backbone_preaugment.diffusion_kind = "vp"
vp_backbone_preaugment.preconditioning = "vp"
vp_backbone_preaugment.block_type = "vp"
vp_backbone_preaugment.fape_weight = 0.0
vp_backbone_preaugment.local_weight = 0.0
vp_backbone_preaugment.encoder_depth = 2
vp_backbone_preaugment.x_weight = 2
vp_backbone_preaugment.pos_weight = 2
vp_backbone_preaugment.rotation_weight = 1
vp_backbone_preaugment.trajectory_weight = 0.5
vp_backbone_preaugment.p_clip = 1.0

vp_backbone_preaugment_tiny = deepcopy(vp_backbone_preaugment)
vp_backbone_preaugment_tiny.local_size = 128

fvp_backbone_preaugment = deepcopy(vp_backbone_preaugment)
fvp_backbone_preaugment.diffusion_time_scale = "framediff"
fvp_backbone_preaugment.diffusion_kind = "vpfixed"
fvp_backbone_preaugment.diffusion_depth = 4
fvp_backbone_preaugment.encoder_depth = 1
fvp_backbone_preaugment.local_size = 128

fvp_backbone_preaugment_noclip = deepcopy(fvp_backbone_preaugment)
fvp_backbone_preaugment_noclip.p_clip = 0.0

flow_backbone_preaugment_noclip = deepcopy(fvp_backbone_preaugment_noclip)
flow_backbone_preaugment_noclip.diffusion_kind = "flow"
flow_backbone_preaugment_noclip.preconditioning = "flow"
flow_backbone_preaugment_noclip.no_trajectory_loss = True
flow_backbone_preaugment_noclip.min_size = 60
flow_backbone_preaugment_noclip.max_size = 384
flow_backbone_preaugment_noclip.x_late = True
flow_backbone_preaugment_noclip.aa_decoder_kind = "adm"
flow_backbone_preaugment_noclip.aa_decoder_depth = 3

vp_backbone_preaugment_pair = deepcopy(vp_backbone_preaugment)
vp_backbone_preaugment_pair.encoder_depth = 2
vp_backbone_preaugment_pair.fape_weight = 0
vp_backbone_preaugment_pair.pair_condition = True

vp_backbone_preaugment_pair_corrcloud = deepcopy(vp_backbone_preaugment_pair)
vp_backbone_preaugment_pair_corrcloud.encoder_depth = 4
vp_backbone_preaugment_pair_corrcloud.correlated_cloud = True

# latent
latent = deepcopy(vp_backbone_preaugment_pair)
latent.pair_size = 64
latent.encoder_depth = 6
latent.diffusion_depth = 8
latent.block_size = 1
latent.latent_size = 20
latent.p_clean = 0.25

hal = deepcopy(vp_backbone_preaugment_pair)
hal.pair_size = 64
hal.encoder_depth = 6
hal.diffusion_depth = 12
hal.block_size = 4
hal.latent_size = 20
hal.p_clean = 0.1

latent_clean = deepcopy(latent)
latent_clean.p_clean = 1.0

simple = deepcopy(latent)
simple.num_iterations = 10

edm_backbone_preaugment_xt = deepcopy(edm_backbone_preaugment)
edm_backbone_preaugment_xt.x_trajectory = True


# AA-loss slingshots back to ln(20) within the first
# ~1000 steps of training.
# This coincides with pos loss stagnating, or also
# slingshotting.
# Therefore, we're testing here if turning off
# the AA loss alleviates pos loss slingshotting.
edm_backbone_preaugment_noaa = deepcopy(edm_backbone_preaugment)
edm_backbone_preaugment_noaa.aa_weight = 0

edm_backbone_augmented_unweighted = deepcopy(edm_backbone_augmented_refine)
edm_backbone_augmented_unweighted.update(dict(
    pos_weight=1.0,
    x_weight=1.0,
    rotation_weight=1.0,
    trajectory_weight=0.5,
    violation_weight=0.01,
    local_weight=1.0,
    aa_weight=1.0,
    fape_weight=0.1,
))

edm_backbone_augmented_norefine = deepcopy(edm_backbone_augmented_simple)
edm_backbone_augmented_norefine.refine = False

minimal_salad = dotdict(
    encoder_depth=4,
    pseudoatom_count=32,
    d_clip=100.0,
    ## feature sizes
    local_size=256,
    pair_size=64,
    ## neighbours (64)
    index_neighbours=16,
    spatial_neighbours=16,
    random_neighbours=32,
    ## module parameters
    pair_vector_features=False,
    factor=4,
    heads=8,
    key_size=32,
    position_scale=10.0,
    ## diffusion stack parameters
    depth=6,
    block_size=1,
    # position diffusion parameters
    sigma_data=10.0,
    ## condition parameters
    pair_condition=False,
    ## output parameters
    # output sidechain positions defined by torsion angles
    head_weights=dotdict(structure=1.0),
)

minimal_salad_stc = deepcopy(minimal_salad)
minimal_salad_stc.scale_to_center = True