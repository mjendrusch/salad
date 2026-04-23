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

hybrid_ve = deepcopy(default_ve_scaled)
hybrid_ve.regularization_weight = 0.01
hybrid_ve.augment_size = 6
hybrid_ve.sidechain_weight = 100.0
hybrid_ve.diffusion_depth = 1
hybrid_ve.aa_weight = 1.0
hybrid_ve.update_mask = "ca"

hybrid_ve_stopE = deepcopy(hybrid_ve)
hybrid_ve_stopE.stop_encoder = True
hybrid_ve_stopE.decoder_atom37 = True

hybrid_ve_atom37 = deepcopy(hybrid_ve_stopE)
hybrid_ve_atom37.encoder_atom37 = True
hybrid_ve_atom37.decoder_atom37 = True

hybrid_ve_enc37 = deepcopy(hybrid_ve_stopE)
hybrid_ve_enc37.encoder_atom37 = True
hybrid_ve_enc37.decoder_atom37 = False

hybrid_ve_strongreg = deepcopy(hybrid_ve_enc37)
hybrid_ve_strongreg.regularization_weight = 0.1

hybrid_ve_norm = deepcopy(hybrid_ve_enc37)
hybrid_ve_norm.normalize_encoder = "rmsd"
hybrid_ve_norm.stop_encoder = False

hybrid_ve_eachnorm = deepcopy(hybrid_ve_norm)
hybrid_ve_eachnorm.normalize_encoder = "length"

hybrid_ve_norm_full = deepcopy(hybrid_ve_norm)
hybrid_ve_norm_full.diffusion_depth = 6

hybrid_ve_norm_unclip = deepcopy(hybrid_ve_norm_full)
hybrid_ve_norm_unclip.clipped_weight = 0.0
hybrid_ve_norm_unclip.unclipped_weight = 1.0

hybrid_ve_norm_fapeclip = deepcopy(hybrid_ve_norm_unclip)
hybrid_ve_norm_fapeclip.fape_clipped = True

hybrid_flow_norm_fapeclip = deepcopy(hybrid_ve_norm_fapeclip)
hybrid_flow_norm_fapeclip.diffusion_kind = "flow"
hybrid_flow_norm_fapeclip.preconditioning = "flow"
hybrid_flow_norm_fapeclip.pos_weight = 2.0
hybrid_flow_norm_fapeclip.trajectory_weight = 1.0

hybrid_flow_norm_noweight = deepcopy(hybrid_flow_norm_fapeclip)
hybrid_flow_norm_noweight.no_loss_weight = True
hybrid_flow_norm_noweight.local_size = 512
hybrid_flow_norm_noweight.pair_size = 128
hybrid_flow_norm_noweight.decoder_depth = 4
hybrid_flow_norm_noweight.no_decoder_stop = True

hybrid_flow_bb = deepcopy(hybrid_flow_norm_noweight)
hybrid_flow_bb.encoder_bb = True
hybrid_flow_bb.encoder_atom37 = False
hybrid_flow_bb.self_condition_decoder = True
hybrid_flow_bb.self_condition_masked = True
hybrid_flow_bb.learned_scale = True
hybrid_flow_bb.local_size = 128
hybrid_flow_bb.pair_size = 64
# hybrid_flow_bb.decoder_bb = True # TODO: how to implement that?
hybrid_flow_bb.clipped_weight = 1.0
hybrid_flow_bb.unclipped_weight = 0.01

hybrid_flow_bb_weight = deepcopy(hybrid_flow_bb)
hybrid_flow_bb_weight.no_loss_weight = False

hybrid_flow_bb_nano = deepcopy(hybrid_flow_bb_weight)
hybrid_flow_bb_nano.depth = 2
hybrid_flow_bb_nano.decoder_depth = 2
hybrid_flow_bb_nano.sidechain_rigid_loss = True

hybrid_flow_bb_full = deepcopy(hybrid_flow_bb_weight)
hybrid_flow_bb_full.local_size = 512
hybrid_flow_bb_full.pair_size = 128
hybrid_flow_bb_full.sidechain_rigid_loss = True
hybrid_flow_bb_full.denoised_rigid_loss = False
hybrid_flow_bb_full.encoder_no_learned_scale = True
hybrid_flow_bb_full.no_override = False
hybrid_flow_bb_full.x_loss_t_weight = False

hybrid_flow_staged = deepcopy(hybrid_flow_bb_full)
hybrid_flow_staged.encoder_no_learned_scale = True
hybrid_flow_staged.unclipped_weight = 0
hybrid_flow_staged.embed_depth = 6
hybrid_flow_staged.refine_depth = 6
hybrid_flow_staged.no_override = True
hybrid_flow_staged.use_dropout = False
hybrid_flow_staged.staged_diffusion = True
hybrid_flow_staged.sidechain_rigid_weight = 100.0
hybrid_flow_staged.internal_recycle = False # TODO: implement this

hybrid_flow_staged_unclip = deepcopy(hybrid_flow_staged)
hybrid_flow_staged_unclip.fape_clipped = True
hybrid_flow_staged_unclip.unclipped_weight = 1.0
hybrid_flow_staged_unclip.clipped_weight = 0.0

hybrid_flow_staged_unclip_reweight = deepcopy(hybrid_flow_staged_unclip)
hybrid_flow_staged_unclip_reweight.uniform_weight = 0.5

hybrid_ve_bb_full = deepcopy(hybrid_flow_bb_full)
hybrid_ve_bb_full.denoised_rigid_loss = False
hybrid_ve_bb_full.preconditioning = "edm_scaled"
hybrid_ve_bb_full.diffusion_kind = "edm"
hybrid_ve_bb_full.pos_weight *= 100
hybrid_ve_bb_full.trajectory_weight *= 100

hybrid_ve_bb_small = deepcopy(hybrid_ve_bb_full)
hybrid_ve_bb_small.local_size = 128
hybrid_ve_bb_small.pair_size = 64

# TODO: hybrid ve atom14

hybrid_flow_sc = deepcopy(hybrid_flow_bb)
hybrid_flow_sc.encoder_bb = False
hybrid_flow_sc.encoder_atom37 = True
hybrid_flow_sc.self_condition_decoder = False
hybrid_flow_sc.local_size = 256
hybrid_flow_sc.pair_size = 128

hybrid_flow_sc_neq = deepcopy(hybrid_flow_sc)
hybrid_flow_sc_neq.non_equivariant = True

hybrid_flow_sc_neq_weight = deepcopy(hybrid_flow_sc_neq)
hybrid_flow_sc_neq_weight.no_loss_weight = False

hybrid_flow_bb_pair_time = deepcopy(hybrid_flow_bb_weight)
hybrid_flow_bb_pair_time.pair_time = True
hybrid_flow_bb_pair_time.invert_pos_mask = True
hybrid_flow_bb_pair_time.fape_weight = 0.0
hybrid_flow_bb_pair_time.fape_trajectory_weight = 0.0

hybrid_flow_bb_decoder_diffusion = deepcopy(hybrid_flow_bb_pair_time)
hybrid_flow_bb_decoder_diffusion.decoder_diffusion = True
hybrid_flow_bb_decoder_diffusion.fape_weight = 1.0

hybrid_flow_sc_protlike = deepcopy(hybrid_flow_bb_weight)
hybrid_flow_sc_protlike.encoder_bb = False
hybrid_flow_sc_protlike.encoder_atom37 = True
hybrid_flow_sc_protlike.pair_time = True
hybrid_flow_sc_protlike.invert_pos_mask = True
hybrid_flow_sc_protlike.encoder_stop = True
hybrid_flow_sc_protlike.decouple_ae = True
hybrid_flow_sc_protlike.no_override = True
hybrid_flow_sc_protlike.fape_weight = 1.0

hybrid_flow_sc_protlike_unclip = deepcopy(hybrid_flow_sc_protlike)
hybrid_flow_sc_protlike_unclip.clipped_weight = 0.0
hybrid_flow_sc_protlike_unclip.unclipped_weight = 1.0
hybrid_flow_sc_protlike_unclip.stop_encoder_loss = True

hybrid_flow_sc_protlike_unclip_nostope = deepcopy(hybrid_flow_sc_protlike_unclip)
hybrid_flow_sc_protlike_unclip_nostope.stop_encoder_loss = False

hybrid_flow_sc_latent = deepcopy(hybrid_flow_sc_protlike)
hybrid_flow_sc_latent.encode_latent = True
hybrid_flow_sc_latent.stop_encoder_loss = True

hybrid_flow_sc_latent_nostop = deepcopy(hybrid_flow_sc_latent)
hybrid_flow_sc_latent_nostop.stop_encoder_loss = False
hybrid_flow_sc_latent_nostop.stop_encoder = False
hybrid_flow_sc_latent_nostop.decouple_ae = False
hybrid_flow_sc_latent_nostop.sidechain_weight = 1.0
hybrid_flow_sc_latent_nostop.clipped_weight = 0.0
hybrid_flow_sc_latent_nostop.unclipped_weight = 1.0
hybrid_flow_sc_latent_nostop.encoder_no_learned_scale = True

hybrid_flow_sc_latent_large = deepcopy(hybrid_flow_sc_latent_nostop)
hybrid_flow_sc_latent_large.local_size = 512
hybrid_flow_sc_latent_large.pair_size = 128
hybrid_flow_sc_latent_large.clipped_weight = 1.0
hybrid_flow_sc_latent_large.unclipped_weight = 0.1
hybrid_flow_sc_latent_large.decouple_ae = True

hybrid_flow_sc_neq_pair_time = deepcopy(hybrid_flow_bb_weight)
hybrid_flow_sc_neq_pair_time.encoder_bb = False
hybrid_flow_sc_neq_pair_time.encoder_atom37 = True
hybrid_flow_sc_neq_pair_time.self_condition_decoder = False
hybrid_flow_sc_neq_pair_time.non_equivariant = True
hybrid_flow_sc_neq_pair_time.pair_time = True
hybrid_flow_sc_neq_pair_time.invert_pos_mask = True
hybrid_flow_sc_neq_pair_time.fape_weight = 1.0
hybrid_flow_sc_neq_pair_time.fape_trajectory_weight = 0.0

hybrid_flow_norm_nostop = deepcopy(hybrid_flow_norm_noweight)
hybrid_flow_norm_nostop.no_decoder_stop = True
hybrid_flow_norm_nostop.local_size = 512
hybrid_flow_norm_nostop.pair_size = 64
hybrid_flow_norm_nostop.clipped_weight = 1.0
hybrid_flow_norm_nostop.unclipped_weight = 0.1

hybrid_ve_norm_softclip = deepcopy(hybrid_ve_norm_unclip)
hybrid_ve_norm_softclip.softclip = True

hybrid_ve_normstop = deepcopy(hybrid_ve_norm)
hybrid_ve_normstop.stop_encoder = True

large_ve_scaled = deepcopy(default_ve_scaled)
large_ve_scaled.local_size = 256
large_ve_scaled.pair_size = 128

large_ve_scaled_neq = deepcopy(large_ve_scaled)
large_ve_scaled_neq.equivariance = "semi_equivariant"
large_ve_scaled_neq.nonequivariant_dense = True
large_ve_scaled_neq.heads = 8
large_ve_scaled_neq.local_size = 128

large_ve_scaled_neq_pair = deepcopy(large_ve_scaled_neq)
large_ve_scaled_neq_pair.use_pair = True

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

large_vp_scaled = deepcopy(default_vp_scaled)
large_vp_scaled.local_size = 256
large_vp_scaled.pair_size = 128

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
