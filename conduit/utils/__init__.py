from typing import Any, List, Tuple
from enum import Enum
from dataclasses import dataclass, is_dataclass, fields

from conduit.compute_provider.runpod.runpod_types import RunpodGpuType


@dataclass(frozen=True)  # TODO impl Topology
class NodeTopology:
    pass


@dataclass
class ComputeOffering:
    id: str
    price_per_hour: int  # cents/hour
    max_available: int
    memory_gb: int
    notes: str
    enterprise_grade: bool  # Runs in certified T3/T4 data centers
    topology: List[NodeTopology] | None = None


def dataclass_to_dict(obj: Any) -> dict:
    """
    Recursively convert a dataclass (or nested dataclass) into
    a nested dict/list/primitive structure.
    """
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = dataclass_to_dict(value)  # recurse
        return result
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def bytes_to_gib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024**3)


def mib_to_gib(num_mib: int | float) -> float:
    return float(num_mib) / 1024


def gib_to_mib(num_gib: int | float) -> float:
    return float(num_gib) * 1024


def gb_to_gib(num_gb: int | float) -> float:
    return float(num_gb) * (10**9) / (1024**3)


def gib_to_bytes(gib: int | float) -> int:
    """GiB -> bytes (binary)."""
    return int(round(float(gib) * (1024**3)))
