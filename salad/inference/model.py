"""Utilities for loading salad models."""

import pickle
import salad.modules.config.noise_schedule_benchmark as config_choices

from salad.modules.noise_schedule_benchmark import (
    StructureDiffusionNoise, StructureDiffusionPredict,
    StructureDiffusionEncode)

def make_salad_model(config, param_path):
    """Get config dictionary and parameters for a salad model.
    
    Args:
        config: name of a salad configuration, e.g. default_vp.
        param_path: path to a corresponding checkpoint.
    """
    config = getattr(config_choices, config)
    with open(param_path, "rb") as f:
        params = pickle.load(f)
    return config, params
