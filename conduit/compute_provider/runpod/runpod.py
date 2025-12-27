import requests
from dataclasses import asdict
from conduit.compute_provider.base import ContainerInfo, ContainerRuntimeProvider
from .runpod_types import PodResponse, PodCreateRequest
from dacite import from_dict


def _drop_nones(obj):
    if isinstance(obj, dict):
        return {k: _drop_nones(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [_drop_nones(v) for v in obj if v is not None]
    return obj


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
        payload = _drop_nones(asdict(input))
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

    def stop_container(self, container_id: str) -> None:
        response = requests.post(
            self.BASE_URL + f"/pods/{container_id}/stop", headers=self.api_headers
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(
                f"POST /pods/{container_id}/stop failed: {response.text}"
            ) from e

    def restart_container(self, container_id: str) -> None:
        response = requests.post(
            self.BASE_URL + f"/pods/{container_id}/start", headers=self.api_headers
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(
                f"POST /pods/{container_id}/start failed: {response.text}"
            ) from e
