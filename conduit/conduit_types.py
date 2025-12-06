from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, List, TypedDict
from collections import defaultdict


class DeploymentStatus(Enum):
    DEPLOYING = "DEPLOYING"
    DEPLOYED = "DEPLOYED"


class NodeStatus(Enum):
    PROVISIONING = "PROVISIONING"
    PROVISIONED = "PROVISIONED"
    DEPLOYED = "DEPLOYED"


class Runtime(Enum):
    VLLM = ("VLLM",)
    LM_LITE = "LM_LITE"
    CUSTOM = "CUSTOM"


class ComputeProvider(Enum):
    RUNPOD = "RUNPOD"


class DeploymentType(Enum):
    LLM = "LLM"
    IMAGE_TO_VIDEO = "I2V"
    IMAGE_TO_IMAGE = "I2I"


class DType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    TF32 = "tf32"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    FP4 = "fp4"


class GPUS(Enum):
    NVIDIA_A100_80GB_PCIE = "NVIDIA A100 80GB PCIe"
    NVIDIA_A100_SXM4_80GB = "NVIDIA A100-SXM4-80GB"
    NVIDIA_A40 = "NVIDIA A40"
    NVIDIA_B200 = "NVIDIA B200"
    NVIDIA_GEFORCE_RTX_3070 = "NVIDIA GeForce RTX 3070"
    NVIDIA_GEFORCE_RTX_3080 = "NVIDIA GeForce RTX 3080"
    NVIDIA_GEFORCE_RTX_3080_TI = "NVIDIA GeForce RTX 3080 Ti"
    NVIDIA_GEFORCE_RTX_3090 = "NVIDIA GeForce RTX 3090"
    NVIDIA_GEFORCE_RTX_3090_TI = "NVIDIA GeForce RTX 3090 Ti"
    NVIDIA_GEFORCE_RTX_4070_TI = "NVIDIA GeForce RTX 4070 Ti"
    NVIDIA_GEFORCE_RTX_4080 = "NVIDIA GeForce RTX 4080"
    NVIDIA_GEFORCE_RTX_4080_SUPER = "NVIDIA GeForce RTX 4080 SUPER"
    NVIDIA_GEFORCE_RTX_4090 = "NVIDIA GeForce RTX 4090"
    NVIDIA_GEFORCE_RTX_5080 = "NVIDIA GeForce RTX 5080"
    NVIDIA_GEFORCE_RTX_5090 = "NVIDIA GeForce RTX 5090"
    NVIDIA_H100_80GB_HBM3 = "NVIDIA H100 80GB HBM3"
    NVIDIA_H100_NVL = "NVIDIA H100 NVL"
    NVIDIA_H100_PCIE = "NVIDIA H100 PCIe"
    NVIDIA_H200 = "NVIDIA H200"
    NVIDIA_L4 = "NVIDIA L4"
    NVIDIA_L40 = "NVIDIA L40"
    NVIDIA_L40S = "NVIDIA L40S"
    NVIDIA_RTX_2000_ADA_GENERATION = "NVIDIA RTX 2000 Ada Generation"
    NVIDIA_RTX_4000_ADA_GENERATION = "NVIDIA RTX 4000 Ada Generation"
    NVIDIA_RTX_4000_SFF_ADA_GENERATION = "NVIDIA RTX 4000 SFF Ada Generation"
    NVIDIA_RTX_5000_ADA_GENERATION = "NVIDIA RTX 5000 Ada Generation"
    NVIDIA_RTX_6000_ADA_GENERATION = "NVIDIA RTX 6000 Ada Generation"
    NVIDIA_RTX_A2000 = "NVIDIA RTX A2000"
    NVIDIA_RTX_A4000 = "NVIDIA RTX A4000"
    NVIDIA_RTX_A4500 = "NVIDIA RTX A4500"
    NVIDIA_RTX_A5000 = "NVIDIA RTX A5000"
    NVIDIA_RTX_A6000 = "NVIDIA RTX A6000"
    NVIDIA_RTX_PRO_6000_BLACKWELL_MAX_Q_WORKSTATION_EDITION = (
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    )
    NVIDIA_RTX_PRO_6000_BLACKWELL_SERVER_EDITION = (
        "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    )
    NVIDIA_RTX_PRO_6000_BLACKWELL_WORKSTATION_EDITION = (
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    )


class Architecture(Enum):
    KEPLER = "Kepler"
    MAXWELL = "Maxwell"
    PASCAL = "Pascal"
    VOLTA = "Volta"
    TURING = "Turing"
    AMPERE_HPC = "Ampere (HPC)"
    AMPERE_CONSUMER = "Ampere (Consumer)"
    ADA_LOVELACE_CONSUMER = "Ada Lovelace (Consumer)"
    ADA_WORKSTATION = "Ada (Workstation)"
    HOPPER = "Hopper"
    BLACKWELL = "Blackwell"


HF_DTYPE_MAP = {
    "float32": DType.FP32,
    "float16": DType.FP16,
    "bfloat16": DType.BF16,
    "tf32": DType.TF32,
    "int8": DType.INT8,
    "int4": DType.INT4,
    "fp8": DType.FP8,
    "fp4": DType.FP4,
}


@dataclass
class ModelConfig:
    id: str
    max_model_concurrency: int = 1
    max_model_len: int = 1024
    model_batch_execute_timeout_ms: int = 500


# 2. Mapping between GPU enum and Architecture enum
ARCHITECTURE_BY_GPU: Dict[GPUS, Architecture] = {
    # Ampere (HPC)
    GPUS.NVIDIA_A100_80GB_PCIE: Architecture.AMPERE_HPC,
    GPUS.NVIDIA_A100_SXM4_80GB: Architecture.AMPERE_HPC,
    # Ampere (Consumer / workstation-class but 8.6)
    GPUS.NVIDIA_A40: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_GEFORCE_RTX_3070: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_GEFORCE_RTX_3080: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_GEFORCE_RTX_3080_TI: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_GEFORCE_RTX_3090: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_GEFORCE_RTX_3090_TI: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_RTX_A2000: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_RTX_A4000: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_RTX_A4500: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_RTX_A5000: Architecture.AMPERE_CONSUMER,
    GPUS.NVIDIA_RTX_A6000: Architecture.AMPERE_CONSUMER,
    # Ada Lovelace (Consumer)
    GPUS.NVIDIA_GEFORCE_RTX_4070_TI: Architecture.ADA_LOVELACE_CONSUMER,
    GPUS.NVIDIA_GEFORCE_RTX_4080: Architecture.ADA_LOVELACE_CONSUMER,
    GPUS.NVIDIA_GEFORCE_RTX_4080_SUPER: Architecture.ADA_LOVELACE_CONSUMER,
    GPUS.NVIDIA_GEFORCE_RTX_4090: Architecture.ADA_LOVELACE_CONSUMER,
    # Ada (Workstation / datacenter)
    GPUS.NVIDIA_L4: Architecture.ADA_WORKSTATION,
    GPUS.NVIDIA_L40: Architecture.ADA_WORKSTATION,
    GPUS.NVIDIA_L40S: Architecture.ADA_WORKSTATION,
    GPUS.NVIDIA_RTX_2000_ADA_GENERATION: Architecture.ADA_WORKSTATION,
    GPUS.NVIDIA_RTX_4000_ADA_GENERATION: Architecture.ADA_WORKSTATION,
    GPUS.NVIDIA_RTX_4000_SFF_ADA_GENERATION: Architecture.ADA_WORKSTATION,
    GPUS.NVIDIA_RTX_5000_ADA_GENERATION: Architecture.ADA_WORKSTATION,
    GPUS.NVIDIA_RTX_6000_ADA_GENERATION: Architecture.ADA_WORKSTATION,
    # Hopper
    GPUS.NVIDIA_H100_80GB_HBM3: Architecture.HOPPER,
    GPUS.NVIDIA_H100_NVL: Architecture.HOPPER,
    GPUS.NVIDIA_H100_PCIE: Architecture.HOPPER,
    GPUS.NVIDIA_H200: Architecture.HOPPER,
    # Blackwell (incl. next-gen GeForce + RTX Pro)
    GPUS.NVIDIA_B200: Architecture.BLACKWELL,
    GPUS.NVIDIA_GEFORCE_RTX_5080: Architecture.BLACKWELL,
    GPUS.NVIDIA_GEFORCE_RTX_5090: Architecture.BLACKWELL,
    GPUS.NVIDIA_RTX_PRO_6000_BLACKWELL_MAX_Q_WORKSTATION_EDITION: Architecture.BLACKWELL,
    GPUS.NVIDIA_RTX_PRO_6000_BLACKWELL_SERVER_EDITION: Architecture.BLACKWELL,
    GPUS.NVIDIA_RTX_PRO_6000_BLACKWELL_WORKSTATION_EDITION: Architecture.BLACKWELL,
}


# Mapping between GPU enum and VRAM in GB
VRAM_BY_GPU: Dict[GPUS, int] = {
    # A-series / Data center
    GPUS.NVIDIA_A100_80GB_PCIE: 80,
    GPUS.NVIDIA_A100_SXM4_80GB: 80,
    GPUS.NVIDIA_A40: 48,
    # B-series
    GPUS.NVIDIA_B200: 180,
    # GeForce RTX 30 series
    GPUS.NVIDIA_GEFORCE_RTX_3070: 8,
    GPUS.NVIDIA_GEFORCE_RTX_3080: 10,
    GPUS.NVIDIA_GEFORCE_RTX_3080_TI: 12,
    GPUS.NVIDIA_GEFORCE_RTX_3090: 24,
    GPUS.NVIDIA_GEFORCE_RTX_3090_TI: 24,
    # GeForce RTX 40 series
    GPUS.NVIDIA_GEFORCE_RTX_4070_TI: 12,
    GPUS.NVIDIA_GEFORCE_RTX_4080: 16,
    GPUS.NVIDIA_GEFORCE_RTX_4080_SUPER: 16,
    GPUS.NVIDIA_GEFORCE_RTX_4090: 24,
    # GeForce RTX 50 series (expected)
    GPUS.NVIDIA_GEFORCE_RTX_5080: 16,
    GPUS.NVIDIA_GEFORCE_RTX_5090: 24,
    # Hopper / Data center
    GPUS.NVIDIA_H100_80GB_HBM3: 80,
    GPUS.NVIDIA_H100_NVL: 94,
    GPUS.NVIDIA_H100_PCIE: 80,
    GPUS.NVIDIA_H200: 141,
    # L-series (inferencing)
    GPUS.NVIDIA_L4: 24,
    GPUS.NVIDIA_L40: 48,
    GPUS.NVIDIA_L40S: 48,
    # Ada Generation (workstation)
    GPUS.NVIDIA_RTX_2000_ADA_GENERATION: 16,
    GPUS.NVIDIA_RTX_4000_ADA_GENERATION: 20,
    GPUS.NVIDIA_RTX_4000_SFF_ADA_GENERATION: 20,
    GPUS.NVIDIA_RTX_5000_ADA_GENERATION: 32,
    GPUS.NVIDIA_RTX_6000_ADA_GENERATION: 48,
    # Older RTX A-series
    GPUS.NVIDIA_RTX_A2000: 6,
    GPUS.NVIDIA_RTX_A4000: 16,
    GPUS.NVIDIA_RTX_A4500: 20,
    GPUS.NVIDIA_RTX_A5000: 24,
    GPUS.NVIDIA_RTX_A6000: 48,
}


DTYPE_TO_ARCHITECTURES = {
    DType.FP32: {
        Architecture.KEPLER,
        Architecture.MAXWELL,
        Architecture.PASCAL,
        Architecture.VOLTA,
        Architecture.TURING,
        Architecture.AMPERE_HPC,
        Architecture.AMPERE_CONSUMER,
        Architecture.ADA_LOVELACE_CONSUMER,
        Architecture.ADA_WORKSTATION,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.FP16: {
        Architecture.KEPLER,  # ⚠️*
        Architecture.MAXWELL,  # ⚠️*
        Architecture.PASCAL,
        Architecture.VOLTA,
        Architecture.TURING,
        Architecture.AMPERE_HPC,
        Architecture.AMPERE_CONSUMER,
        Architecture.ADA_LOVELACE_CONSUMER,
        Architecture.ADA_WORKSTATION,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.BF16: {
        Architecture.AMPERE_HPC,
        Architecture.AMPERE_CONSUMER,
        Architecture.ADA_LOVELACE_CONSUMER,
        Architecture.ADA_WORKSTATION,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.TF32: {
        Architecture.AMPERE_HPC,
        Architecture.AMPERE_CONSUMER,
        Architecture.ADA_LOVELACE_CONSUMER,
        Architecture.ADA_WORKSTATION,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.INT8: {
        Architecture.VOLTA,
        Architecture.TURING,
        Architecture.AMPERE_HPC,
        Architecture.AMPERE_CONSUMER,
        Architecture.ADA_LOVELACE_CONSUMER,
        Architecture.ADA_WORKSTATION,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.INT4: {
        Architecture.TURING,  # ⚠️
        Architecture.AMPERE_HPC,
        Architecture.AMPERE_CONSUMER,  # ⚠️
        Architecture.ADA_LOVELACE_CONSUMER,
        Architecture.ADA_WORKSTATION,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.FP8: {
        Architecture.ADA_LOVELACE_CONSUMER,
        Architecture.ADA_WORKSTATION,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.FP4: {
        Architecture.AMPERE_HPC,  # ⚠️
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
}


def bytes_per_dtype(dtype: DType) -> float:
    return {
        DType.FP32: 4.0,
        DType.TF32: 4.0,
        DType.FP16: 2.0,
        DType.BF16: 2.0,
        DType.FP8: 1.0,
        DType.INT8: 1.0,
        DType.FP4: 0.5,
        DType.INT4: 0.5,
    }[dtype]


def architectures_for_dtype(dtype: DType):
    try:
        return DTYPE_TO_ARCHITECTURES[dtype]
    except KeyError:
        raise ValueError(f"No architecture support information for dtype: {dtype}")


def architecture_from_gpu(gpu: GPUS) -> Architecture:
    """
    Given a GPUS enum value, return its Architecture.
    """
    try:
        return ARCHITECTURE_BY_GPU[gpu]
    except KeyError as e:
        raise ValueError(f"No architecture mapping defined for GPU: {gpu}") from e


GPUS_BY_ARCHITECTURE: Dict[Architecture, List[GPUS]] = defaultdict(list)
for gpu, arch in ARCHITECTURE_BY_GPU.items():
    GPUS_BY_ARCHITECTURE[arch].append(gpu)


def gpus_from_architecture(arch: Architecture) -> List[GPUS]:
    """
    Given an Architecture, return all GPU enum values that use it.
    """
    return list(GPUS_BY_ARCHITECTURE.get(arch, []))


def architecture_from_gpu_name(name: str) -> Architecture:
    """
    Look up architecture starting from the display string of the GPU.
    """
    gpu_enum = GPUS(name)
    return architecture_from_gpu(gpu_enum)


def map_hf_dtype(dtype: str) -> DType:
    try:
        return HF_DTYPE_MAP[dtype.lower()]
    except KeyError:
        raise ValueError(f"Unsupported HF dtype: {dtype}")


def get_vram_for_gpu(gpu: GPUS) -> int:
    """
    Returns VRAM (in GB) corresponding to the given GPU enum.

    Raises:
        ValueError: if the GPU is not found in the VRAM map.
    """
    try:
        return VRAM_BY_GPU[gpu]
    except KeyError:
        raise ValueError(f"GPU {gpu} is not registered in VRAM_BY_GPU")
