import os
import time
import math
import ipaddress
from typing import Any, Dict, TypedDict, List
from conduit.compute_provider.local import (
    LocalContainerCreateRequest,
    LocalNetworkBinding,
    LocalProvider,
    LocalProviderOverrides,
)
from conduit.conduit_types import ComputeProvider
from conduit.compute_provider.base import ContainerInfo, ContainerRuntimeProvider
from conduit.compute_provider.runpod import (
    PodResponse,
    Runpod,
    PodCreateRequest,
    VRAM_GB_BY_GPU_ID as RUNPOD_VRAM_GB_BY_GPU_ID,
)
from conduit.compute_provider.runpod.utils import (
    get_gpus as runpod_get_gpus,
    runpod_gpu_type_to_compute_offerings,
)
from conduit.utils import ComputeOffering
from conduit.utils.accelerators.nvidia import GPUHostingResult
from conduit.utils.deployment import DeploymentConstraint


from .base import EnvConfig

CONTAINER_PROVIDER_PROVISION_TIMEOUT_SEC = 60
CONTAINER_PROVISION_CHECK_TIMEOUT = 5


def start_container_provision(
    provider: ComputeProvider,
    hosting_result: GPUHostingResult,
    image: str,
    ports: str,
    gpu_type: str,
    num_gpus: int,
    env: EnvConfig,
    container_size_gb: float,
    compute_provider_config_overrides: dict,
) -> str:
    compute_provider = get_compute_provider(provider)
    container_start_params = get_compute_provider_start_params(
        provider,
        hosting_result,
        image,
        ports,
        gpu_type,
        num_gpus,
        env,
        container_size_gb,
        compute_provider_config_overrides,
    )
    container_start_response = compute_provider.start_container(container_start_params)
    return compute_provider.serialize_start_external_id(container_start_response)


def stop_provider_node(provider: ComputeProvider, external_id: str):
    compute_provider = get_compute_provider(provider)
    compute_provider.stop_container(external_id)
    return external_id


def restart_provider_node(provider: ComputeProvider, external_id: str):
    compute_provider = get_compute_provider(provider)
    compute_provider.restart_container(external_id)
    container_info = wait_node_provision(provider, external_id)
    return compute_provider.serialize_start_external_id(container_info)


def wait_node_provision(
    provider: ComputeProvider,
    container_external_id: str,
) -> ContainerInfo:

    compute_provider = get_compute_provider(provider)

    while True:
        time.sleep(CONTAINER_PROVISION_CHECK_TIMEOUT)
        response = compute_provider.get_container(container_external_id)
        if provider_provisioned_resource(provider, response):
            break

    return compute_provider.serialize_create_response(response)


def deprovision(provider: ComputeProvider, container_identifer: str):
    match provider:
        case ComputeProvider.RUNPOD:
            compute_provider = get_compute_provider(provider)
            compute_provider.terminate_container(container_identifer)

        case ComputeProvider.LOCAL:
            compute_provider = get_compute_provider(provider)
            compute_provider.terminate_container(container_identifer)
        case _:
            raise ValueError(f"Unknown compute provider: {provider}")


def get_compute_provider(provider: ComputeProvider) -> ContainerRuntimeProvider:
    match provider:
        case ComputeProvider.RUNPOD:
            token = os.environ.get("RUNPOD_API_KEY")
            if not token:
                raise RuntimeError("RUNPOD_API_KEY not in environment")
            return Runpod(token)

        case ComputeProvider.LOCAL:
            return LocalProvider()
        case _:
            raise ValueError(f"Unknown compute provider: {provider}")


def get_compute_provider_start_params(
    provider: ComputeProvider,
    host_result: GPUHostingResult,
    container_image: str,
    ports: str,
    gpu_type: str,
    num_gpus: int,
    env: EnvConfig,
    container_size_gb: float,
    overrides: dict | None = None,
) -> Any:
    overrides = overrides or {}
    match provider:
        case ComputeProvider.RUNPOD:
            ports_list = [f"{p}/tcp" for p in ports.split(",")]
            cloud_type = (
                "COMMUNITY"
                if not host_result.compute_offering.enterprise_grade
                else "SECURE"
            )
            base_config = {
                "imageName": container_image,
                "containerDiskInGb": math.ceil(container_size_gb),
                "gpuTypeIds": [gpu_type],
                "cloudType": cloud_type,
                "gpuCount": num_gpus,
                "ports": ports_list,
                "env": env["env"],
            }

            pod_config = base_config | overrides
            return PodCreateRequest(**pod_config)
        case ComputeProvider.LOCAL:
            port_map = (
                {f"{p.strip()}/tcp": int(p.strip()) for p in ports.split(",")}
                if ports
                else None
            )
            binding = overrides.get("binding", LocalNetworkBinding.LOCAL)
            return LocalContainerCreateRequest(
                image=container_image,
                env=env,
                binding=binding,
                port_map=port_map,
            )
        case _:
            raise ValueError(f"Unknown compute provider: {provider}")


def provider_provisioned_resource(provider: ComputeProvider, response: Any) -> bool:
    match provider:
        case ComputeProvider.RUNPOD:
            try:
                ip_obj = ipaddress.ip_address(response.publicIp)
                return isinstance(ip_obj, ipaddress.IPv4Address)
            except Exception:
                return False
        case ComputeProvider.LOCAL:
            ip_obj = ipaddress.ip_address(response.publicIp)
            return isinstance(ip_obj, ipaddress.IPv4Address)
    return False


def get_provider_compute_offerings(provider: ComputeProvider) -> List[ComputeOffering]:
    match provider:
        case ComputeProvider.RUNPOD:
            gpus = runpod_get_gpus()
            compute_offerings = [
                offering
                for gpu in gpus
                for offering in runpod_gpu_type_to_compute_offerings(gpu)
            ]
            return compute_offerings

    raise ValueError("Compute provider not supporte")


def get_compute_provider_gpu_map(provider: ComputeProvider) -> dict[str, int]:
    match provider:
        case ComputeProvider.RUNPOD:
            return RUNPOD_VRAM_GB_BY_GPU_ID

    raise ValueError("Compute provider not supported")
