import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Sequence, Mapping
from conduit.utils import ComputeOffering
from conduit.utils.model import DType


class Architecture(Enum):
    KEPLER = "Kepler"
    MAXWELL = "Maxwell"
    PASCAL = "Pascal"
    VOLTA = "Volta"
    TURING = "Turing"
    AMPERE = "Ampere"
    ADA = "Ada Lovelace"
    HOPPER = "Hopper"
    BLACKWELL = "Blackwell"
    UNKNOWN = "UNKNOWN"


@dataclass
class NvidiaGPU:
    name: str
    gpu_count: int
    memory_mib: int  # Total VRAM (MiB, binary)
    memory_used_mib: int  # Used VRAM (MiB, binary)
    memory_free_mib: int  # Free VRAM (MiB, binary)
    architecture: Architecture


@dataclass
class GPUHostingResult:
    gpus: NvidiaGPU
    required_vram_gb: float
    raw_model_vram_gb: float
    total_capacity_gb: float
    headroom_gb: float
    can_host: bool
    price_cents: int
    compute_offering: ComputeOffering | None = None


ARCH_PATTERNS = [
    (
        Architecture.BLACKWELL,
        [
            r"\bB(?:100|200)\b",
            r"\bGB200\b",
            r"\bRTX\W*50\d{2}\b(?!\W*Ada)",
            r"\bRTX\s+PRO\s+6000\b.*\bBlackwell\b",
            r"\bBlackwell\b",
        ],
    ),
    # --- Hopper ---
    (
        Architecture.HOPPER,
        [
            r"\bH100\b",
            r"\bH200\b",
            r"\bGH200\b",
            r"\bHopper\b",
        ],
    ),
    # --- Ada Lovelace (consumer + workstation + datacenter) ---
    (
        Architecture.ADA,
        [
            # datacenter
            r"\bL4\b",
            r"\bL40S?\b",
            # workstation branding
            r"\bRTX\s+\d{4}\s+Ada(?:\s+Generation)?\b",  # RTX 2000/4000/5000/6000 Ada Generation
            r"\bRTX\s+\d{4}\s+SFF\s+Ada(?:\s+Generation)?\b",
            # consumer branding
            r"\b(RTX\s*)?40\d{2}\b",  # 4070/4080/4090 etc
            r"\bAda(?:\s+Lovelace)?\b",
        ],
    ),
    # --- Ampere ---
    (
        Architecture.AMPERE,
        [
            # datacenter
            r"\bA100\b",
            r"\bA30\b",
            r"\bA40\b",
            # workstation (RTX A2000/A4000/A4500/A5000/A6000…)
            r"\bRTX\s+A\d{4}\b",
            # consumer
            r"\b(RTX\s*)?30\d{2}\b",
            r"\bAmpere\b",
        ],
    ),
    # --- Turing ---
    (
        Architecture.TURING,
        [
            r"\b(RTX\s*)?20\d{2}\b(?!\s*Ada)",  # don't steal "RTX 2000 Ada"
            r"\bGTX\s*16\d{2}\b",
            r"\bT4\b",
            r"\bTuring\b",
        ],
    ),
    # --- Volta ---
    (
        Architecture.VOLTA,
        [
            r"\b(Tesla\s*)?V100\b",
            r"\bGV100\b",
            r"\bVolta\b",
        ],
    ),
    # --- Pascal ---
    (
        Architecture.PASCAL,
        [
            r"\bGTX\s*10\d{2}\b",
            r"\b(Tesla\s*)?P(?:4|40|100)\b",
            r"\bQuadro\s+P\d{3,4}\b",
            r"\bPascal\b",
        ],
    ),
    # --- Maxwell ---
    (
        Architecture.MAXWELL,
        [
            r"\b(GT\s*7[0-4]\d|GTX\s*7[0-4]\d|GM\d+)\b",
            r"\bGTX\s*9(?:50|60|70|80)\b",
            r"\bMaxwell\b",
        ],
    ),
    # --- Kepler ---
    (
        Architecture.KEPLER,
        [
            r"\b(GT\s*6\d{2}|GTX\s*6\d{2}|K[0-9]+)\b",
            r"\bKepler\b",
        ],
    ),
]

DTYPE_TO_NVIDIA_ARCHITECTURES = {
    DType.FP32: {
        Architecture.KEPLER,
        Architecture.MAXWELL,
        Architecture.PASCAL,
        Architecture.VOLTA,
        Architecture.TURING,
        Architecture.AMPERE,
        Architecture.ADA,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.FP16: {
        Architecture.PASCAL,
        Architecture.VOLTA,
        Architecture.TURING,
        Architecture.AMPERE,
        Architecture.ADA,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.BF16: {
        Architecture.AMPERE,
        Architecture.ADA,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.TF32: {
        Architecture.AMPERE,
        Architecture.ADA,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.INT8: {
        Architecture.PASCAL,
        Architecture.VOLTA,
        Architecture.TURING,
        Architecture.AMPERE,
        Architecture.ADA,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.INT4: {
        Architecture.TURING,
        Architecture.AMPERE,
        Architecture.ADA,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.FP8: {
        Architecture.ADA,
        Architecture.HOPPER,
        Architecture.BLACKWELL,
    },
    DType.FP4: {
        Architecture.BLACKWELL,
    },
}


def architectures_for_dtype(dtype: DType):
    try:
        return DTYPE_TO_NVIDIA_ARCHITECTURES[dtype]
    except KeyError:
        raise ValueError(f"No architecture support information for dtype: {dtype}")


def guess_architecture_regex(gpu_name: str) -> Architecture:
    if not gpu_name:
        return None
    name = gpu_name.strip().lower()

    for arch, patterns in ARCH_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return arch

    return Architecture.UNKNOWN


def detect_nvidia() -> NvidiaGPU | None:
    def _parse_memory_mib(s: str) -> int:
        s = s.replace("MiB", "").strip()
        try:
            return int(float(s))
        except ValueError:
            return 0

    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=name,memory.total,memory.used "
            "--format=csv,noheader",
            shell=True,
            stderr=subprocess.DEVNULL,
        ).decode(errors="ignore")
    except Exception:
        return None

    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return None

    gpu_count = len(lines)

    # assume homogeneous GPUs
    parts = [p.strip() for p in lines[0].split(",")]
    name = parts[0]
    total_mib = _parse_memory_mib(parts[1])
    used_mib = sum(_parse_memory_mib(l.split(",")[2]) for l in lines)
    free_mib = gpu_count * total_mib - used_mib

    return NvidiaGPU(
        name=name,
        gpu_count=gpu_count,
        memory_mib=gpu_count * total_mib,
        memory_used_mib=used_mib,
        memory_free_mib=free_mib,
        architecture=guess_architecture_regex(name),
    )


def build_nvidia_gpus(
    gpu_names: Sequence[str],
    vram_gb_by_id: Mapping[str, int],
    *,
    strict: bool = True,
) -> list[NvidiaGPU]:
    out: list[NvidiaGPU] = []

    for idx, name in enumerate(gpu_names):
        gb = vram_gb_by_id.get(name)
        if gb is None:
            if strict:
                raise KeyError(f"Missing VRAM mapping for GPU id: {name!r}")
            continue

        total_mib = int(gb * 1024)  # GB-as-GiB -> MiB
        out.append(
            NvidiaGPU(
                name=name,
                gpu_count=1,
                memory_mib=total_mib,
                memory_used_mib=0,
                memory_free_mib=total_mib,
                architecture=guess_architecture_regex(name),
            )
        )

    return out


def build_nvidia_gpus_from_compute_offering(
    offerings: Sequence["ComputeOffering"],
    *,
    strict: bool = True,
) -> list["NvidiaGPU"]:
    out: list["NvidiaGPU"] = []

    for offering in offerings:
        gb = getattr(offering, "memory_gb", None)

        if gb is None:
            if strict:
                raise KeyError(
                    f"Missing memory_gb for ComputeOffering id: {offering.id!r}"
                )
            continue

        total_mib = int(gb * 1024)  # GB-as-GiB -> MiB

        out.append(
            NvidiaGPU(
                gpu_count=1,
                name=offering.id,  # name=id in this context
                memory_mib=total_mib,
                memory_used_mib=0,
                memory_free_mib=total_mib,
                architecture=guess_architecture_regex(offering.id),
            )
        )

    return out
