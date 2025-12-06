from abc import ABC, abstractmethod
from typing import Dict

from dataclasses import dataclass


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
    def start_container(self) -> str:  # id of created resource
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
