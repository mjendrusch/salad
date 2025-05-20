
import pickle
import salad.modules.config.noise_schedule_benchmark as config_choices

from salad.modules.noise_schedule_benchmark import (
    StructureDiffusionNoise, StructureDiffusionPredict)

def make_salad_model(config, param_path):
    config = getattr(config_choices, config)
    with open(param_path, "rb") as f:
            params = pickle.load(f)
    return config, params
