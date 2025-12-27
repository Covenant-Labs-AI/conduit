from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, List, TypedDict
from collections import defaultdict


class DeploymentStatus(Enum):
    DEPLOYING = "DEPLOYING"
    DEPLOYED = "DEPLOYED"
    STOPPED = "STOPPED"


class NodeStatus(Enum):
    PROVISIONING = "PROVISIONING"
    PROVISIONED = "PROVISIONED"
    DEPLOYED = "DEPLOYED"
    STOPPED = "STOPPED"


class Runtime(Enum):
    VLLM = ("VLLM",)
    LM_LITE = "LM_LITE"
    CUSTOM = "CUSTOM"


class ComputeProvider(Enum):
    LOCAL = "LOCAL"
    RUNPOD = "RUNPOD"


class DeploymentType(Enum):
    LLM = "LLM"


@dataclass
class LmModelConfig:
    id: str
    max_model_len: int = 1024
    max_model_concurrency: int = 1


@dataclass
class LmLiteModelConfig(LmModelConfig):
    model_batch_execute_timeout_ms: int = 500
