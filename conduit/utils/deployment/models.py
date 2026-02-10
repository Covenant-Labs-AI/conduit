import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from conduit.utils import ComputeOffering
from conduit.utils.accelerators.nvidia import NvidiaGPU

if TYPE_CHECKING:
    from conduit.runtime import LMLiteBlock, VLLMBlock


class DeploymentConstraint(Enum):
    ENTERPRISE = "ENTERPRISE"
    SINGLE_DEVICE = "SINGLE_DEVICE"
    HIGH_BANDWIDTH_INTERCONNECT = "HIGH_BANDWIDTH_INTERCONNECT"


@dataclass
class ComputeOfferingCandidate:
    offering: ComputeOffering
    gpu: NvidiaGPU
    num_gpu: int
    price: int


@dataclass(frozen=True)
class LLMVramProfile:
    max_position_embeddings: int

    dtype: str
    quant_dtype: str | None

    num_hidden_layers: int
    num_attention_heads: int | None
    num_kv_heads: int
    head_dim: int
    hidden_size: int | None


def gpu_id_and_count_label(gpus: NvidiaGPU, sep: str = ", ") -> tuple[str, str, int]:
    if not gpus:
        return ("", "", 0)

    gpu_id = gpus.name
    num_gpus = gpus.gpu_count
    display = f"{gpu_id}x {num_gpus}"
    return (gpu_id, display, num_gpus)


def compute_deployment_key(block: "LMLiteBlock | VLLMBlock") -> str:
    payload = {
        "runtime": block.runtime.value,
        "image": block.image,
        "provider": block.compute_provider.value,
        "constraints": block.constraints,
        "replicas": block.replicas,
        "models": [
            {
                "id": m.id,
                "max_model_len": m.max_model_len,
            }
            for m in block.models
        ],
        "overrides": block.compute_provider_config_overrides or {},
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        default=lambda o: o.value if isinstance(o, Enum) else str(o),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
