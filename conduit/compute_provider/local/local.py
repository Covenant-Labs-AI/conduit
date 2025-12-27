import os
import docker
from docker import errors
from docker.types import DeviceRequest
from dataclasses import dataclass
from typing import Dict, List, Any, TypedDict
from enum import Enum
from conduit.compute_provider.base import (
    ContainerRuntimeProvider,
    EnvConfig,
    ContainerInfo,
)


class LocalNetworkBinding(Enum):
    LOCAL = "127.0.0.1"
    PUBLIC = "0.0.0.0"


class LocalProviderOverrides(TypedDict, total=False):
    binding: LocalNetworkBinding


@dataclass
class LocalContainerCreateRequest:
    image: str
    env: EnvConfig | None = None
    binding: LocalNetworkBinding = LocalNetworkBinding.LOCAL
    port_map: Dict[str, int] | None = None


@dataclass
class LocalContainerInfo:
    id: str
    publicIp: str
    name: str
    status: str
    image: List[str]
    ports: Dict[str, Any]
    env: List[str]
    mounts: List[Dict[str, Any]]
    created: str
    labels: Dict[str, str]
    command: str | None


class LocalProvider(ContainerRuntimeProvider):
    def __init__(self):
        try:
            self.client = docker.from_env()
        except errors.DockerException as e:
            raise RuntimeError(
                "Docker is not reachable. Make sure the daemon is running and you have permission to access the Docker socket "
                "(e.g., add your user to the 'docker' group and relogin)."
            ) from e

    @staticmethod
    def _normalize_ports_and_public_ip(
        attrs: Dict[str, Any]
    ) -> tuple[Dict[str, int], str | None]:
        raw_ports = attrs.get("NetworkSettings", {}).get("Ports", {}) or {}
        ports: Dict[str, int] = {}
        public_ip: str | None = None

        for key, bindings in raw_ports.items():
            if not bindings:
                continue

            host_binding = bindings[0]
            host_port_str = host_binding.get("HostPort")
            if not host_port_str:
                continue

            container_port = key.split("/")[0]

            ports[container_port] = int(host_port_str)

            if public_ip is None:
                public_ip = host_binding.get("HostIp")

        return ports, public_ip

    def get_container(self, id_or_name: str) -> LocalContainerInfo:
        container = self.client.containers.get(id_or_name)
        attrs = container.attrs

        ports, public_ip = self._normalize_ports_and_public_ip(attrs)

        return LocalContainerInfo(
            id=container.id,
            name=container.name,
            status=container.status,
            image=container.image.tags,
            ports=ports,
            env=attrs["Config"].get("Env", []),
            mounts=attrs.get("Mounts", []),
            created=attrs.get("Created", ""),
            labels=attrs["Config"].get("Labels", {}),
            command=attrs["Config"].get("Cmd"),
            publicIp=public_ip,
        )

    def restart_container(self, id_or_name: str) -> LocalContainerInfo:
        container = self.client.containers.get(id_or_name)
        container.reload()

        if container.status != "running":
            container.start()

        container.reload()
        attrs = container.attrs

        ports, public_ip = self._normalize_ports_and_public_ip(attrs)

        return LocalContainerInfo(
            id=container.id,
            name=container.name,
            status=container.status,
            image=container.image.tags,
            ports=ports,
            env=attrs["Config"].get("Env", []),
            mounts=attrs.get("Mounts", []),
            created=attrs.get("Created", ""),
            labels=attrs["Config"].get("Labels", {}),
            command=attrs["Config"].get("Cmd"),
            publicIp=public_ip,
        )

    def stop_container(self, id_or_name: str) -> LocalContainerInfo:
        container = self.client.containers.get(id_or_name)
        container.reload()

        if container.status == "running":
            container.stop(timeout=5)

        container.reload()
        attrs = container.attrs

        ports, public_ip = self._normalize_ports_and_public_ip(attrs)

        return LocalContainerInfo(
            id=container.id,
            name=container.name,
            status=container.status,
            image=container.image.tags,
            ports=ports,
            env=attrs["Config"].get("Env", []),
            mounts=attrs.get("Mounts", []),
            created=attrs.get("Created", ""),
            labels=attrs["Config"].get("Labels", {}),
            command=attrs["Config"].get("Cmd"),
            publicIp=public_ip,
        )

    def start_container(
        self, container_config: LocalContainerCreateRequest
    ) -> LocalContainerInfo:
        image = self.client.images.pull(container_config.image)
        ports = None

        if container_config.port_map:
            binding_ip = container_config.binding.value
            ports = {
                container_port: (binding_ip, host_port)
                for container_port, host_port in container_config.port_map.items()
            }

        device_requests = [
            DeviceRequest(
                count=-1,
                capabilities=[["gpu"]],
            )
        ]

        container = self.client.containers.run(
            image,
            detach=True,
            environment=container_config.env.get("env", {}),
            ports=ports,
            device_requests=device_requests,
            volumes={
                os.getcwd(): {"bind": "/workspace", "mode": "rw"},
            },
        )

        container.reload()
        attrs = container.attrs

        clean_ports, _ = self._normalize_ports_and_public_ip(attrs)

        return LocalContainerInfo(
            id=container.id,
            publicIp=container_config.binding.value,
            name=container.name,
            status=container.status,
            image=container.image.tags,
            ports=clean_ports,
            env=attrs["Config"].get("Env", []),
            mounts=attrs.get("Mounts", []),
            created=attrs.get("Created", ""),
            labels=attrs["Config"].get("Labels", {}),
            command=attrs["Config"].get("Cmd"),
        )

    def terminate_container(self, id_or_name: str):
        try:
            container = self.client.containers.get(id_or_name)

            if container.status == "running":
                container.stop(timeout=5)

            container.remove(force=True)

        except errors.NotFound:
            pass

    @classmethod
    def serialize_create_response(cls, response: LocalContainerInfo) -> "ContainerInfo":
        return ContainerInfo(
            id=response.id, port_map=response.ports, public_ip=response.publicIp
        )

    @classmethod
    def serialize_start_external_id(cls, response: LocalContainerInfo) -> str:
        return response.id


__all__ = [
    "LocalNetworkBinding",
    "LocalProviderOverrides",
    "LocalContainerCreateRequest",
    "LocalContainerInfo",
    "LocalProvider",
]
