import json
import os
import requests
from typing import List

from conduit.compute_provider.runpod.runpod_types import RunpodGpuType
from conduit.utils import ComputeOffering

API_URL = "https://api.runpod.io/graphql"
QUERY_GPU_TYPES = """
query GpuTypes {
  gpuTypes {
    id
    secureCloud
    communityCloud
    maxGpuCount
    maxGpuCountCommunityCloud
    maxGpuCountSecureCloud
    securePrice
    memoryInGb
    communityPrice
    minPodGpuCount
    nodeGroupGpuSizes
    nodeGroupDatacenters { id }
  }
}
"""


def runpod_gpu_type_to_compute_offerings(gpu: RunpodGpuType) -> List[ComputeOffering]:
    offerings: List[ComputeOffering] = []
    if gpu.secureCloud:
        offerings.append(
            ComputeOffering(
                id=gpu.id,
                price_per_hour=int(round(gpu.securePrice * 100)),
                memory_gb=gpu.memoryInGb,
                max_available=gpu.maxGpuCountSecureCloud,
                notes="secure cloud",
                enterprise_grade=True,
            )
        )
    if gpu.communityCloud:
        offerings.append(
            ComputeOffering(
                id=gpu.id,
                price_per_hour=int(round(gpu.communityPrice * 100)),
                memory_gb=gpu.memoryInGb,
                max_available=gpu.maxGpuCountCommunityCloud,
                notes="community cloud",
                enterprise_grade=False,
            )
        )

    return offerings


def get_gpus(timeout: float = 30.0) -> List[RunpodGpuType]:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("Missing RUNPOD_API_KEY environment variable.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(
        API_URL,
        headers=headers,
        data=json.dumps({"query": QUERY_GPU_TYPES}),
        timeout=timeout,
    )
    resp.raise_for_status()

    payload = resp.json()

    if payload.get("errors"):
        message = payload["errors"][0].get("message", "Unknown GraphQL error")
        raise RuntimeError(f"GraphQL error: {message}")

    items = payload["data"]["gpuTypes"]

    return [RunpodGpuType(**item) for item in items]
