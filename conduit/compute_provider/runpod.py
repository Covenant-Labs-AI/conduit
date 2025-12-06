import requests
from enum import Enum
from dataclasses import asdict
from dataclasses import dataclass, field, asdict
from conduit.compute_provider.base import ContainerInfo, ContainerRuntimeProvider
from typing import Dict, List, Mapping, Optional, Literal, Any
from dacite import from_dict


def drop_nones(obj):
    if isinstance(obj, dict):
        return {k: drop_nones(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [drop_nones(v) for v in obj if v is not None]
    return obj


class Category(str, Enum):
    NVIDIA = "NVIDIA"
    AMD = "AMD"
    CPU = "CPU"


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


# ---- Nested dataclasses ----


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


class Runpod(ContainerRuntimeProvider):
    BASE_URL = "https://rest.runpod.io/v1"

    def __init__(self, token: str):
        self.token = token
        self.api_headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @classmethod
    def serialize_create_response(cls, create_response: PodResponse) -> ContainerInfo:
        return ContainerInfo(
            id=create_response.id,
            public_ip=create_response.publicIp,
            port_map=create_response.portMappings,
        )

    @classmethod
    def serialize_start_external_id(cls, create_response: PodResponse) -> str:
        return create_response.id

    def get_container(self, container_id: str) -> PodResponse:
        response = requests.get(
            self.BASE_URL + f"/pods/{container_id}", headers=self.api_headers
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"GET /pods/{container_id} failed: {response.text}") from e
        pod = from_dict(data_class=PodResponse, data=response.json())
        return pod

    def start_container(self, input: PodCreateRequest) -> PodResponse:
        payload = drop_nones(asdict(input))
        response = requests.post(
            self.BASE_URL + "/pods", headers=self.api_headers, json=payload
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"POST /pods failed: {response.text}") from e
        pod = from_dict(data_class=PodResponse, data=response.json())
        return pod

    def terminate_container(self, container_id: str) -> None:
        response = requests.delete(
            self.BASE_URL + f"/pods/{container_id}", headers=self.api_headers
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(
                f"DELETE /pods/{container_id} failed: {response.text}"
            ) from e


### Contract Tests
def test_template_deploy():
    import os
    import time

    runpod_client = Runpod(token=os.environ.get("RUNPOD_API_KEY"))
    input = PodCreateRequest(
        templateId="lfzit4klmi", gpuTypeIds=["NVIDIA A100-SXM4-80GB"]
    )
    container = runpod_client.start_container(input=input)
    while True:
        time.sleep(10)
        container = runpod_client.get_container(container.id)
        time.sleep(10)
        runpod_client.terminate_container(container.id)
        break

def test_image_deploy():
    import os
    import time

    runpod_client = Runpod(token=os.environ.get("RUNPOD_API_KEY"))
    input = PodCreateRequest(
        imageName="registry.covenantlabs.ai/runlet-0.10.0:latest",
        containerRegistryAuthId="cmgty2b930001jx02krfd4c5o",
        gpuTypeIds=["NVIDIA A100-SXM4-80GB"],
    )
    container = runpod_client.start_container(input=input)
    while True:
        time.sleep(10)
        container = runpod_client.get_container(container.id)
        print(container)
        time.sleep(10)
        runpod_client.terminate_container(container.id)
        break
