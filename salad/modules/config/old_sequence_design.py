from salad.modules.utils.collections import dotdict, deepcopy

tiny = dotdict(
    local_size=128,
    pair_size=128,
    norm="orb",
    block_type="extended",
    ca_only=True,
    depth=3,
    block_size=1,
    pos_noise=0.3,
    num_neighbours=32,
    num_rbf=64,
    l_max=8,
    heads=8,
    key_size=32,
    key_points=4,
    factor=4,
    pair_attention=True,
    environment="orb",
    potts="linear",
    losses=dotdict(
        aa=1.0,
        aa_pseudo=1.0,
        aa_ar=1.0,
        aa_sc=0.0,
        pssm_term=0.01,
        contact_term=0.0,
    )
)

medium = deepcopy(tiny)
medium.depth = 6
medium.num_neighbours = 32
medium.lmax = 3

medium_resi = deepcopy(medium)
medium_resi.resi_pair_features = True
medium_resi.local_env = True
medium_resi.norm = "post"

medium_full = deepcopy(medium_resi)
medium_full.environment = "orb"
medium_full.ca_only = False
medium_full.learn_smol_env = False
medium_full.label_smoothing = 0.1
medium_full.num_rbf = 16
medium_full.l_max = 3

medium_nrn = deepcopy(medium_full)
medium_nrn.no_renorm = True
medium_nrn.num_neighbours = 48

adm = deepcopy(medium_nrn)
adm.depth = 3
adm.decoder_depth = 3
adm.potts = "caliby"
adm.decoders = dotdict(
    potts=dotdict(aa=1.0, aa_pseudo_log_p=1.0),
    adm=dotdict(aa=1.0)
)

medium_caliby = deepcopy(medium_nrn)
medium_caliby.potts = "caliby"
medium_caliby.losses = dotdict(
    aa=1.0,
    aa_pseudo=0.0,
    aa_ar=0.0,
    aa_sc=0.0,
    aa_pseudo_log_p=1.0,
    aa_ar_log_p=1.0,
    pssm_term=0.0,
    contact_term=0.0,
)

medium_dist = deepcopy(medium)
medium_dist.environment = "distance"

medium_mpnn = deepcopy(medium)
medium_mpnn.environment = "distance"
medium_mpnn.block_type = "mpnn"
medium_mpnn.pair_attention = False
medium_mpnn.losses.aa_sc = 0.0

default = deepcopy(tiny)
default.depth = 6
default.num_neighbours = 48
default.factor = 2
default.norm = "post"
default.embed_pos = False
default.pair_attention = False

default_attn = deepcopy(default)
default_attn.norm = "post"
default_attn.embed_pos = False
default_attn.pair_attention = True
