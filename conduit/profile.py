from typing import Dict

from conduit.conduit_types import GPUS

# TODO will have accurate vram profiling
MODEL_VRAM_PROFILE = {}


def get_model_vram_profile(model_id: str) -> Dict[GPUS, dict] | None:
    return MODEL_VRAM_PROFILE.get(model_id)
