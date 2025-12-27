from abc import ABC, abstractmethod
from typing import Dict, TypedDict

from dataclasses import dataclass


class EnvConfig(TypedDict):
    env: Dict[str, str]


@dataclass
class ContainerInfo:
    id: str
    public_ip: str
    port_map: Dict[str, int] | None


class ContainerRuntimeProvider(ABC):
    @abstractmethod
    def get_container(self):
        pass

    @abstractmethod
    def start_container(self) -> str:
        pass

    @abstractmethod
    def restart_container(self):
        pass

    @abstractmethod
    def stop_container(self):
        pass

    @abstractmethod
    def terminate_container(self):
        pass

    @classmethod
    @abstractmethod
    def serialize_start_external_id(cls, response) -> str:
        pass

    @classmethod
    @abstractmethod
    def serialize_create_response(cls, response) -> "ContainerInfo":
        """
        Convert provider-specific response object into ContainerInfo
        """
        pass
