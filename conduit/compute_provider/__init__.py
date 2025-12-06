import os
import time
import math
import ipaddress
from typing import Any, Dict, TypedDict
from conduit.conduit_types import GPUS, ComputeProvider
from conduit.compute_provider.base import ContainerInfo, ContainerRuntimeProvider
from conduit.compute_provider.runpod import PodResponse, Runpod, PodCreateRequest


CONTAINER_PROVIDER_PROVISION_TIMEOUT_SEC = 60
CONTAINER_PROVISION_CHECK_TIMEOUT = 5


class EnvConfig(TypedDict):
    env: Dict[str, str]


def start_container_provision(
    provider: ComputeProvider,
    image: str,
    ports: str,
    gpu_type: GPUS,
    env: EnvConfig,
    container_size_gb: float,
    compute_provider_config_overrides: dict,
) -> str:
    compute_provider = get_compute_provider(provider)
    container_start_params = get_compute_provider_start_params(
        provider,
        image,
        ports,
        gpu_type,
        env,
        container_size_gb,
        compute_provider_config_overrides,
    )
    # TODO better typing abstract create reqs into an adapter method
    container_start_response = compute_provider.start_container(container_start_params)
    return compute_provider.serialize_start_external_id(container_start_response)


def wait_node_provision(
    provider: ComputeProvider,
    container_external_id: str,
) -> ContainerInfo:

    compute_provider = get_compute_provider(provider)

    while True:
        time.sleep(CONTAINER_PROVISION_CHECK_TIMEOUT)
        #  need to wait for the container to provision
        response = compute_provider.get_container(container_external_id)
        if provider_provisioned_resource(provider, response):
            break

    return compute_provider.serialize_create_response(response)


def deprovision(provider: ComputeProvider, container_identifer: str):
    match provider:
        case ComputeProvider.RUNPOD:
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

        case _:
            raise ValueError(f"Unknown compute provider: {provider}")


def get_compute_provider_start_params(
    provider: ComputeProvider,
    container_image: str,
    ports: str,
    gpu_type: GPUS,
    env: EnvConfig,
    container_size_gb: float,
    overrides: dict | None = None,
) -> str:
    overrides = overrides or {}
    match provider:
        case ComputeProvider.RUNPOD:
            ports_list = [f"{p}/tcp" for p in ports.split(",")]
            base_config = {
                "imageName": container_image,
                "containerDiskInGb": math.ceil(container_size_gb),
                "gpuTypeIds": [gpu_type.value],
                "ports": ports_list,
                "env": env["env"],
            }

            pod_config = base_config | overrides
            return PodCreateRequest(**pod_config)
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
    return False
