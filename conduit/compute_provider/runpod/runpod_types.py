from enum import Enum
from dataclasses import dataclass, field
from typing import Literal, List, Dict, Final, Annotated, Optional, Any


class Category(str, Enum):
    NVIDIA = "NVIDIA"
    AMD = "AMD"
    CPU = "CPU"


@dataclass(frozen=True)
class RunpodGpuType:
    id: str
    secureCloud: bool
    communityCloud: bool
    maxGpuCount: int
    maxGpuCountCommunityCloud: int
    maxGpuCountSecureCloud: int
    securePrice: float
    memoryInGb: int
    communityPrice: float
    minPodGpuCount: Optional[int]
    nodeGroupGpuSizes: Optional[
        List[int]
    ]  # this seems to only work for the A100,H200, H100's
    nodeGroupDatacenters: Optional[List[Dict[str, Any]]]


# ---------- Enums (as Literals) ----------

CudaVersion = Literal[
    "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8"
]

CloudType = Literal["SECURE", "COMMUNITY"]
ComputeType = Literal["GPU", "CPU"]
Priority = Literal["availability", "custom"]

CpuFlavorId = Literal["cpu3c", "cpu3g", "cpu3m", "cpu5c", "cpu5g", "cpu5m"]

DataCenterId = Literal[
    "EU-RO-1",
    "CA-MTL-1",
    "EU-SE-1",
    "US-IL-1",
    "EUR-IS-1",
    "EU-CZ-1",
    "US-TX-3",
    "EUR-IS-2",
    "US-KS-2",
    "US-GA-2",
    "US-WA-1",
    "US-TX-1",
    "CA-MTL-3",
    "EU-NL-1",
    "US-TX-4",
    "US-CA-2",
    "US-NC-1",
    "OC-AU-1",
    "US-DE-1",
    "EUR-IS-3",
    "CA-MTL-2",
    "AP-JP-1",
    "EUR-NO-1",
    "EU-FR-1",
    "US-KS-3",
    "US-GA-1",
]

GpuTypeId = Literal[
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A40",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4500",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA L4",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA RTX A4000",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA H200",
    "NVIDIA L40",
    "NVIDIA H100 NVL",
    "NVIDIA H100 PCIe",
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3070",
    "Tesla V100-PCIE-16GB",
    "AMD Instinct MI300X OAM",
    "NVIDIA RTX A2000",
    "Tesla V100-FHHL-16GB",
    "NVIDIA GeForce RTX 4080 SUPER",
    "Tesla V100-SXM2-16GB",
    "NVIDIA GeForce RTX 4070 Ti",
    "Tesla V100-SXM2-32GB",
    "NVIDIA RTX 4000 SFF Ada Generation",
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA A30",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA GeForce RTX 5080",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA B200",
]


class GPUS:
    RTX_4090: Final[Annotated[Literal["NVIDIA GeForce RTX 4090"], "VRAM: 24 GB"]] = (
        "NVIDIA GeForce RTX 4090"
    )
    A40: Final[Annotated[Literal["NVIDIA A40"], "VRAM: 48 GB"]] = "NVIDIA A40"
    RTX_A5000: Final[Annotated[Literal["NVIDIA RTX A5000"], "VRAM: 24 GB"]] = (
        "NVIDIA RTX A5000"
    )
    RTX_3090: Final[Annotated[Literal["NVIDIA GeForce RTX 3090"], "VRAM: 24 GB"]] = (
        "NVIDIA GeForce RTX 3090"
    )
    RTX_A4500: Final[Annotated[Literal["NVIDIA RTX A4500"], "VRAM: 20 GB"]] = (
        "NVIDIA RTX A4500"
    )
    RTX_A6000: Final[Annotated[Literal["NVIDIA RTX A6000"], "VRAM: 48 GB"]] = (
        "NVIDIA RTX A6000"
    )
    L40S: Final[Annotated[Literal["NVIDIA L40S"], "VRAM: 48 GB"]] = "NVIDIA L40S"
    L4: Final[Annotated[Literal["NVIDIA L4"], "VRAM: 24 GB"]] = "NVIDIA L4"
    H100_80GB_HBM3: Final[
        Annotated[Literal["NVIDIA H100 80GB HBM3"], "VRAM: 80 GB"]
    ] = "NVIDIA H100 80GB HBM3"
    RTX_4000_ADA: Final[
        Annotated[Literal["NVIDIA RTX 4000 Ada Generation"], "VRAM: 20 GB"]
    ] = "NVIDIA RTX 4000 Ada Generation"
    A100_80GB_PCIE: Final[
        Annotated[Literal["NVIDIA A100 80GB PCIe"], "VRAM: 80 GB"]
    ] = "NVIDIA A100 80GB PCIe"
    A100_SXM4_80GB: Final[
        Annotated[Literal["NVIDIA A100-SXM4-80GB"], "VRAM: 80 GB"]
    ] = "NVIDIA A100-SXM4-80GB"
    RTX_A4000: Final[Annotated[Literal["NVIDIA RTX A4000"], "VRAM: 16 GB"]] = (
        "NVIDIA RTX A4000"
    )
    RTX_6000_ADA: Final[
        Annotated[Literal["NVIDIA RTX 6000 Ada Generation"], "VRAM: 48 GB"]
    ] = "NVIDIA RTX 6000 Ada Generation"
    RTX_2000_ADA: Final[
        Annotated[Literal["NVIDIA RTX 2000 Ada Generation"], "VRAM: 16 GB"]
    ] = "NVIDIA RTX 2000 Ada Generation"
    H200: Final[Annotated[Literal["NVIDIA H200"], "VRAM: 141 GB"]] = "NVIDIA H200"
    L40: Final[Annotated[Literal["NVIDIA L40"], "VRAM: 48 GB"]] = "NVIDIA L40"
    H100_NVL: Final[Annotated[Literal["NVIDIA H100 NVL"], "VRAM: 94 GB"]] = (
        "NVIDIA H100 NVL"
    )
    H100_PCIE: Final[Annotated[Literal["NVIDIA H100 PCIe"], "VRAM: 80 GB"]] = (
        "NVIDIA H100 PCIe"
    )
    RTX_3080_TI: Final[
        Annotated[Literal["NVIDIA GeForce RTX 3080 Ti"], "VRAM: 12 GB"]
    ] = "NVIDIA GeForce RTX 3080 Ti"
    RTX_3080: Final[Annotated[Literal["NVIDIA GeForce RTX 3080"], "VRAM: 10 GB"]] = (
        "NVIDIA GeForce RTX 3080"
    )
    RTX_3070: Final[Annotated[Literal["NVIDIA GeForce RTX 3070"], "VRAM: 8 GB"]] = (
        "NVIDIA GeForce RTX 3070"
    )
    V100_PCIE_16GB: Final[Annotated[Literal["Tesla V100-PCIE-16GB"], "VRAM: 16 GB"]] = (
        "Tesla V100-PCIE-16GB"
    )
    MI300X_OAM: Final[Annotated[Literal["AMD Instinct MI300X OAM"], "VRAM: 192 GB"]] = (
        "AMD Instinct MI300X OAM"
    )
    RTX_A2000: Final[Annotated[Literal["NVIDIA RTX A2000"], "VRAM: 12 GB"]] = (
        "NVIDIA RTX A2000"
    )
    V100_FHHL_16GB: Final[Annotated[Literal["Tesla V100-FHHL-16GB"], "VRAM: 16 GB"]] = (
        "Tesla V100-FHHL-16GB"
    )
    RTX_4080_SUPER: Final[
        Annotated[Literal["NVIDIA GeForce RTX 4080 SUPER"], "VRAM: 16 GB"]
    ] = "NVIDIA GeForce RTX 4080 SUPER"
    V100_SXM2_16GB: Final[Annotated[Literal["Tesla V100-SXM2-16GB"], "VRAM: 16 GB"]] = (
        "Tesla V100-SXM2-16GB"
    )
    RTX_4070_TI: Final[
        Annotated[Literal["NVIDIA GeForce RTX 4070 Ti"], "VRAM: 12 GB"]
    ] = "NVIDIA GeForce RTX 4070 Ti"
    V100_SXM2_32GB: Final[Annotated[Literal["Tesla V100-SXM2-32GB"], "VRAM: 32 GB"]] = (
        "Tesla V100-SXM2-32GB"
    )
    RTX_4000_SFF_ADA: Final[
        Annotated[Literal["NVIDIA RTX 4000 SFF Ada Generation"], "VRAM: 20 GB"]
    ] = "NVIDIA RTX 4000 SFF Ada Generation"
    RTX_5000_ADA: Final[
        Annotated[Literal["NVIDIA RTX 5000 Ada Generation"], "VRAM: 32 GB"]
    ] = "NVIDIA RTX 5000 Ada Generation"
    RTX_5090: Final[Annotated[Literal["NVIDIA GeForce RTX 5090"], "VRAM: 32 GB"]] = (
        "NVIDIA GeForce RTX 5090"
    )
    A30: Final[Annotated[Literal["NVIDIA A30"], "VRAM: 24 GB"]] = "NVIDIA A30"
    RTX_4080: Final[Annotated[Literal["NVIDIA GeForce RTX 4080"], "VRAM: 16 GB"]] = (
        "NVIDIA GeForce RTX 4080"
    )
    RTX_5080: Final[Annotated[Literal["NVIDIA GeForce RTX 5080"], "VRAM: 16 GB"]] = (
        "NVIDIA GeForce RTX 5080"
    )
    RTX_3090_TI: Final[
        Annotated[Literal["NVIDIA GeForce RTX 3090 Ti"], "VRAM: 24 GB"]
    ] = "NVIDIA GeForce RTX 3090 Ti"
    B200: Final[Annotated[Literal["NVIDIA B200"], "VRAM: 180 GB"]] = "NVIDIA B200"


VRAM_GB_BY_GPU_ID: dict[str, int] = {
    "NVIDIA GeForce RTX 4090": 24,
    "NVIDIA A40": 48,
    "NVIDIA RTX A5000": 24,
    "NVIDIA GeForce RTX 3090": 24,
    "NVIDIA RTX A4500": 20,
    "NVIDIA RTX A6000": 48,
    "NVIDIA L40S": 48,
    "NVIDIA L4": 24,
    "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA RTX 4000 Ada Generation": 20,
    "NVIDIA A100 80GB PCIe": 80,
    "NVIDIA A100-SXM4-80GB": 80,
    "NVIDIA RTX A4000": 16,
    "NVIDIA RTX 6000 Ada Generation": 48,
    "NVIDIA RTX 2000 Ada Generation": 16,
    "NVIDIA H200": 141,
    "NVIDIA L40": 48,
    "NVIDIA H100 NVL": 94,
    "NVIDIA H100 PCIe": 80,
    "NVIDIA GeForce RTX 3080 Ti": 12,
    "NVIDIA GeForce RTX 3080": 10,
    "NVIDIA GeForce RTX 3070": 8,
    "Tesla V100-PCIE-16GB": 16,
    "AMD Instinct MI300X OAM": 192,
    "NVIDIA RTX A2000": 12,
    "Tesla V100-FHHL-16GB": 16,
    "NVIDIA GeForce RTX 4080 SUPER": 16,
    "Tesla V100-SXM2-16GB": 16,
    "NVIDIA GeForce RTX 4070 Ti": 12,
    "Tesla V100-SXM2-32GB": 32,
    "NVIDIA RTX 4000 SFF Ada Generation": 20,
    "NVIDIA RTX 5000 Ada Generation": 32,
    "NVIDIA GeForce RTX 5090": 32,
    "NVIDIA A30": 24,
    "NVIDIA GeForce RTX 4080": 16,
    "NVIDIA GeForce RTX 5080": 16,
    "NVIDIA GeForce RTX 3090 Ti": 24,
    "NVIDIA B200": 180,
}


DesiredStatus = Literal["RUNNING", "EXITED", "TERMINATED"]


# ---------- Request: Pod creation input ----------


@dataclass
class PodCreateRequest:
    # General placement / cloud
    cloudType: CloudType = "SECURE"
    computeType: ComputeType = "GPU"

    # Container / image
    imageName: str = ""
    dockerEntrypoint: Optional[List[str]] = field(default_factory=list)
    dockerStartCmd: Optional[List[str]] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None

    # Resources
    containerDiskInGb: Optional[int] = 50
    volumeInGb: Optional[int] = 20
    volumeMountPath: str = "/workspace"
    networkVolumeId: Optional[str] = None

    # GPU-related
    gpuCount: int = 1
    gpuTypeIds: Optional[List[GpuTypeId]] = field(default_factory=list)
    gpuTypePriority: Priority = "availability"
    minRAMPerGPU: int = 8
    minVCPUPerGPU: int = 2
    allowedCudaVersions: Optional[List[CudaVersion]] = field(default_factory=list)

    # CPU-related (used when computeType == "CPU")
    vcpuCount: Optional[int] = 2
    cpuFlavorIds: Optional[List[CpuFlavorId]] = field(default_factory=list)
    cpuFlavorPriority: Priority = "availability"

    # Networking
    globalNetworking: bool = False
    ports: Optional[List[str]] = field(
        default_factory=list
    )  # e.g. ["8888/http", "22/tcp"]
    supportPublicIp: Optional[bool] = None
    countryCodes: Optional[List[str]] = field(default_factory=list)

    # Location preferences
    dataCenterIds: Optional[List[DataCenterId]] = None
    dataCenterPriority: Priority = "availability"

    # Other controls
    interruptible: bool = False
    locked: bool = False
    name: str = "conduit pod"
    templateId: Optional[str] = None

    # Minimum machine performance
    minDiskBandwidthMBps: Optional[float] = None
    minDownloadMbps: Optional[float] = None
    minUploadMbps: Optional[float] = None

    # Auth
    containerRegistryAuthId: Optional[str] = None

    def __post_init__(self) -> None:
        # Basic validations aligned with docs
        if self.computeType == "GPU":
            if self.gpuCount is None or self.gpuCount < 1:
                raise ValueError("gpuCount must be >= 1 for GPU Pods.")
            # CPU-only fields are ignored by API when GPU is chosen, but we allow them to be set.
        else:  # CPU
            # GPU-only fields are ignored by API when CPU is chosen; keep minimal sanity.
            if self.vcpuCount is not None and self.vcpuCount < 1:
                raise ValueError("vcpuCount must be >= 1 for CPU Pods.")
        if self.ports:
            for p in self.ports:
                if "/" not in p:
                    raise ValueError(
                        f"Port '{p}' must be formatted as '<number>/<protocol>'."
                    )


@dataclass
class GPUInfo:
    id: Optional[str]
    count: Optional[int]
    displayName: Optional[str]
    securePrice: Optional[float]
    communityPrice: Optional[float]
    oneMonthPrice: Optional[float]
    threeMonthPrice: Optional[float]
    sixMonthPrice: Optional[float]
    oneWeekPrice: Optional[float]
    communitySpotPrice: Optional[float]
    secureSpotPrice: Optional[float]


@dataclass
class CPUTypeInfo:
    id: Optional[str]
    displayName: Optional[str]
    cores: Optional[int]
    threadsPerCore: Optional[int]
    groupId: Optional[str]


@dataclass
class MachineInfo:
    minPodGpuCount: Optional[int]
    gpuTypeId: Optional[str]
    gpuType: Optional[GPUInfo]
    cpuCount: Optional[int]
    cpuTypeId: Optional[str]
    cpuType: Optional[CPUTypeInfo]
    location: Optional[str]
    dataCenterId: Optional[str]
    diskThroughputMBps: Optional[int]
    maxDownloadSpeedMbps: Optional[int]
    maxUploadSpeedMbps: Optional[int]
    supportPublicIp: Optional[bool]
    secureCloud: Optional[bool]
    maintenanceStart: Optional[str]
    maintenanceEnd: Optional[str]
    maintenanceNote: Optional[str]
    note: Optional[str]
    costPerHr: Optional[float]
    currentPricePerGpu: Optional[float]
    gpuAvailable: Optional[int]
    gpuDisplayName: Optional[str]


@dataclass
class NetworkVolumeInfo:
    id: Optional[str]
    name: Optional[str]
    size: Optional[int]
    dataCenterId: Optional[str]


@dataclass
class SavingsPlanInfo:
    costPerHr: float
    endTime: str
    gpuTypeId: str
    id: str
    podId: str
    startTime: str


@dataclass
class PodResponse:
    adjustedCostPerHr: Optional[float] = None
    aiApiId: Optional[str] = None
    consumerUserId: Optional[str] = None
    containerDiskInGb: Optional[int] = None
    containerRegistryAuthId: Optional[str] = None
    costPerHr: Optional[float] = None
    cpuFlavorId: Optional[str] = None
    desiredStatus: Optional[str] = None

    endpointId: Optional[str] = None
    env: Optional[Dict[str, str]] = field(default_factory=dict)
    id: Optional[str] = None
    imageName: Optional[str] = None
    lastStartedAt: Optional[str] = None
    lastStatusChange: Optional[str] = None
    locked: Optional[bool] = None
    machine: Optional["MachineInfo"] = None
    machineId: Optional[str] = None
    memoryInGb: Optional[int] = None
    name: Optional[str] = None
    networkVolume: Optional["NetworkVolumeInfo"] = None
    portMappings: Optional[Dict[str, int]] = field(default_factory=dict)
    ports: Optional[List[str]] = field(default_factory=list)
    publicIp: Optional[str] = None
    savingsPlans: Optional[List["SavingsPlanInfo"]] = field(default_factory=list)
    slsVersion: Optional[int] = None
    templateId: Optional[str] = None
    vcpuCount: Optional[int] = None
    volumeEncrypted: Optional[bool] = None
    volumeInGb: Optional[int] = None
    volumeMountPath: Optional[str] = None
    interruptible: Optional[bool] = None
    gpu: Optional["GPUInfo"] = None
    dockerEntrypoint: Optional[List[str]] = field(default_factory=list)
    dockerStartCmd: Optional[List[str]] = field(default_factory=list)


__all__ = [
    # enums / constants
    "Category",
    "CudaVersion",
    "CloudType",
    "ComputeType",
    "Priority",
    "CpuFlavorId",
    "DataCenterId",
    "GpuTypeId",
    "GPUS",
    "VRAM_GB_BY_GPU_ID",
    "DesiredStatus",
    "PodCreateRequest",
    "GPUInfo",
    "CPUTypeInfo",
    "MachineInfo",
    "NetworkVolumeInfo",
    "SavingsPlanInfo",
    "PodResponse",
]
