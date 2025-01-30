from salad.modules.utils.collections import dotdict, deepcopy
import numpy as np

default = dotdict(
    ## feature sizes
    local_size=128,
    pair_size=64,
    latent_size=20,
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
    num_recycle=0,
    ## aa diffusion parameters
    # no sequence diffusion
    aa_decoder_depth=3,
    encoder_depth=3,
    ## loss parameters
    # loss clipping
    p_clip=1.0,
    clip_fape=100,
    # time embedding
    time_embedding=False,
    # local loss parameters
    local_neighbours=16,
    fape_neighbours=64,
    # loss weights
    local_weight=1.0,
    aa_weight=10.0,
    fape_weight=1,
    fape_trajectory_weight=0.5,
    # dataset constraints
    min_size=50,
    max_size=None
)

small = deepcopy(default)
small.aa_decoder_depth = 1
small.encoder_depth = 1
small.distogram_block = "mlp"

small_inner = deepcopy(small)
small_inner.distogram_block = "inner"

small_inner_noise = deepcopy(small_inner)
small_inner_noise.noise_encoder = 0.3

small_inner_input_diffusion = deepcopy(small_inner)
small_inner_input_diffusion.encoder_depth = 2
small_inner_input_diffusion.input_diffusion = True
small_inner_input_diffusion.latent_loss_scale = 0
small_inner_input_diffusion.violation_scale = 1.0
small_inner_input_diffusion.clip_fape = np.inf

small_inner_latent_diffusion = deepcopy(small_inner)
small_inner_latent_diffusion.encoder_depth = 2
small_inner_latent_diffusion.latent_diffusion = True
small_inner_latent_diffusion.latent_loss_scale = 1.0
small_inner_latent_diffusion.time_embedding = True
small_inner_latent_diffusion.violation_scale = 1.0
small_inner_latent_diffusion.clip_fape = 100.0
small_inner_latent_diffusion.unclipped_weight = None

small_inner_latent_diffusion_kabsch = deepcopy(small_inner)
small_inner_latent_diffusion_kabsch.encoder_depth = 2
small_inner_latent_diffusion_kabsch.latent_diffusion = True
small_inner_latent_diffusion_kabsch.latent_loss_scale = 1.0
small_inner_latent_diffusion_kabsch.time_embedding = True # FIXME
small_inner_latent_diffusion_kabsch.violation_scale = 0.1
small_inner_latent_diffusion_kabsch.clip_fape = 100.0
small_inner_latent_diffusion_kabsch.unclipped_weight = None
small_inner_latent_diffusion_kabsch.kabsch_rmsd = None

small_inner_latent_diffusion_noembed = deepcopy(small_inner)
small_inner_latent_diffusion_noembed.encoder_depth = 4
small_inner_latent_diffusion_noembed.latent_diffusion = True
small_inner_latent_diffusion_noembed.latent_loss_scale = 1.0
small_inner_latent_diffusion_noembed.time_embedding = True # FIXME
small_inner_latent_diffusion_noembed.violation_scale = 0.1
small_inner_latent_diffusion_noembed.clip_fape = 100.0
small_inner_latent_diffusion_noembed.noembed = True
small_inner_latent_diffusion_noembed.unclipped_weight = None
small_inner_latent_diffusion_noembed.kabsch_rmsd = None

small_inner_latent_diffusion_noembed_vp = deepcopy(small_inner)
small_inner_latent_diffusion_noembed_vp.encoder_depth = 4
small_inner_latent_diffusion_noembed_vp.latent_diffusion = True
small_inner_latent_diffusion_noembed_vp.vp_diffusion = True
small_inner_latent_diffusion_noembed_vp.latent_loss_scale = 10.0
small_inner_latent_diffusion_noembed_vp.time_embedding = False
small_inner_latent_diffusion_noembed_vp.violation_scale = 0.1
small_inner_latent_diffusion_noembed_vp.clip_fape = 100.0
small_inner_latent_diffusion_noembed_vp.noembed = True
small_inner_latent_diffusion_noembed_vp.unclipped_weight = None
small_inner_latent_diffusion_noembed_vp.kabsch_rmsd = None

small_inner_input_diffusion_time = deepcopy(small_inner)
small_inner_input_diffusion_time.encoder_depth = 2
small_inner_input_diffusion_time.input_diffusion = True
small_inner_input_diffusion_time.latent_loss_scale = 0
small_inner_input_diffusion_time.time_embedding = True
small_inner_input_diffusion_time.violation_scale = 1.0
small_inner_input_diffusion_time.clip_fape = 100.0
small_inner_input_diffusion_time.unclipped_weight = 0.1

small_inner_input_diffusion_4 = deepcopy(small_inner)
small_inner_input_diffusion_4.encoder_depth = 4
small_inner_input_diffusion_4.input_diffusion = True
small_inner_input_diffusion_4.latent_loss_scale = 0
small_inner_input_diffusion_4.violation_scale = 0.01

small_inner_input_diffusion_latent128_4 = deepcopy(small_inner)
small_inner_input_diffusion_latent128_4.encoder_depth = 4
small_inner_input_diffusion_latent128_4.input_diffusion = True
small_inner_input_diffusion_latent128_4.latent_loss_scale = 0
small_inner_input_diffusion_latent128_4.latent_size = 128
small_inner_input_diffusion_latent128_4.violation_scale = 0.01

small_inner_latent_diffusion = deepcopy(small_inner)
small_inner_latent_diffusion.encoder_depth = 2
small_inner_latent_diffusion.latent_diffusion = True
small_inner_latent_diffusion.latent_loss_scale = 1.0
small_inner_latent_diffusion.violation_scale = 0.01

small_nodist_vq = deepcopy(small)
small_nodist_vq.codebook_size = 4096
small_nodist_vq.codebook_b = 0.25
small_nodist_vq.codebook_loss_scale = 1.0
small_nodist_vq.distogram_block = "none"

small_vq = deepcopy(small)
small_vq.distogram_block = "inner"
small_vq.codebook_size = 4096
small_vq.codebook_loss_scale = 1.0
small_vq.codebook_b = 0.25

small_fsq = deepcopy(small)
small_fsq.distogram_block = "inner"
small_fsq.fsq = True

small_vq_e2 = deepcopy(small_vq)
small_vq_e2.encoder_depth = 2

# hard codebook loss scaling to reduce gradient error
small_vq_e2_hard = deepcopy(small_vq_e2)
small_vq_e2_hard.codebook_loss_scale = 10.0

small_vq_e2_affine = deepcopy(small_vq)
small_vq_e2_affine.encoder_depth = 2
small_vq_e2_affine.affine = True

small_vq_decoder = deepcopy(small_vq)
small_vq_decoder.is_decoder = True
small_vq_decoder.depth = 12
small_vq_decoder.param_path = "/g/korbel/mjendrusch/runs/debug-salad-new/salad/sae-small_vq-1024-debug-sae-vquniform-4/checkpoint-200000.jax"

small_vqsmall = deepcopy(small_vq)
small_vqsmall.codebook_size = 1024

small_vqstate = deepcopy(small_vq)
small_vqstate.state = True

small_nonequivariant = deepcopy(small)
small_nonequivariant.distogram_block = "inner"
small_nonequivariant.equivariance = "nonequivariant"

small_semiequivariant = deepcopy(small)
small_semiequivariant.distogram_block = "inner"
small_semiequivariant.equivariance = "semiequivariant"

small_semiequivariant_vq = deepcopy(small_semiequivariant)
small_semiequivariant_vq.codebook_size = 4096
small_semiequivariant_vq.codebook_b = 0.25
small_semiequivariant_vq.codebook_loss_scale = 1.0

small_semiequivariant_vq_e2 = deepcopy(small_semiequivariant_vq)
small_semiequivariant_vq_e2.encoder_depth = 2

small_none = deepcopy(small)
small_none.distogram_block = "none"
