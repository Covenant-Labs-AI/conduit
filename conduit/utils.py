import json
import hashlib
from dataclasses import is_dataclass, asdict, fields
from typing import TYPE_CHECKING, Any, Dict, Optional, List, Type, TypeVar
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from conduit.conduit_types import (
    ModelConfig,
    bytes_per_dtype,
    DType,
    get_vram_for_gpu,
    map_hf_dtype,
    GPUS,
    VRAM_BY_GPU,
)
from conduit.profile import get_model_vram_profile


TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

if TYPE_CHECKING:  # lol? better pattern for typing circ imports?
    from conduit.runtime import LMLiteBlock


def parse_llm_json(json_response: str, output_type: Type[TOut]) -> TOut:
    """
    Safely deserialize an LLM JSON response into the provided dataclass type.

    If the response comes from a "thinking" model and includes a </think> tag,
    everything up to and including </think> is stripped before parsing.

    Args:
        json_response: The raw JSON string from the LLM (possibly with <think>...</think>).
        output_type: The dataclass type expected as output.

    Returns:
        An instance of `output_type`.

    Raises:
        ValueError: If JSON is invalid or missing required fields.
        TypeError: If output_type is not a dataclass.
    """
    if not is_dataclass(output_type):
        raise TypeError("output_type must be a dataclass type")

    think_end = "</think>"  # if thinking model
    if think_end in json_response:
        json_response = json_response.split(think_end, 1)[1].strip()

    try:
        data = json.loads(json_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON from LLM: {e.msg}\n\nFull response:\n{json_response}"
        ) from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object, got {type(data).__name__}")

    output_fields = {f.name for f in fields(output_type)}
    missing = output_fields - set(data.keys())

    if missing:
        raise ValueError(
            f"Missing expected fields in LLM response: {', '.join(sorted(missing))}"
        )

    try:
        return output_type(**{f: data[f] for f in output_fields})  # type: ignore[arg-type]
    except TypeError as e:
        raise ValueError(f"LLM response contained wrong field types: {e}") from e


def compute_deployment_key(block: "LMLiteBlock") -> str:
    payload = {
        "runtime": block.runtime.value,
        "image": block.image,
        "provider": block.compute_provider.value,
        "gpu": block.gpu.value if block.gpu else None,
        "replicas": block.replicas,
        "models": [
            {
                "id": m.id,
                "max_model_concurrency": m.max_model_concurrency,
                "max_model_len": m.max_model_len,
            }
            for m in block.models
        ],
        "overrides": block.compute_provider_config_overrides or {},
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_model_index_and_config(
    repo_id: str, revision: str | None = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Downloads model.safetensors.index.json and config.json independently if they exist.
    Returns a dict with:
        - "index": parsed JSON or None
        - "config": parsed JSON or None
    Raises FileNotFoundError if *neither* file exists.
    """
    result = {"index": None, "config": None}

    for key, filename in {
        "index": "model.safetensors.index.json",
        "config": "config.json",
    }.items():
        try:
            path = hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision
            )
            with open(path, "r", encoding="utf-8") as f:
                result[key] = json.load(f)
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                continue
            raise

    if result["index"] is None and result["config"] is None:
        raise FileNotFoundError(
            f"Neither 'model.safetensors.index.json' nor 'config.json' found in repo '{repo_id}'"
        )

    return result


def get_single_model_size(repo_id: str) -> int:
    """
    Return the size in bytes of the single model file in a HF repo,
    assuming there is NO model index and only one weight file.
    """
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)

    # info.siblings contains metadata for all files in the repo
    weight_exts = (".bin", ".safetensors", ".h5", ".msgpack")

    weight_files = [s for s in info.siblings if s.rfilename.endswith(weight_exts)]

    if any(s.rfilename.endswith("model_index.json") for s in info.siblings):
        raise RuntimeError("Repo appears sharded (model_index.json present).")

    if len(weight_files) == 0:
        raise FileNotFoundError("No model weight file found in repo.")
    if len(weight_files) > 1:
        raise RuntimeError(
            "Multiple model files found but no model index; not sure which to pick."
        )

    # size is in bytes
    size_bytes = weight_files[0].size
    return size_bytes


def calculate_container_size_gb(models: List[ModelConfig], runtime_size_gb=0) -> int:
    total_size = 0
    for model in models:
        results = load_model_index_and_config(model.id)
        config = results.get("config")  # model config
        model_index = results.get("index")  # model index

        if model_index:
            size: int = model_index["metadata"]["total_size"]
        else:
            size = get_single_model_size(model.id)
        total_size += size

    return total_size / (1024**3) + runtime_size_gb


def best_gpu_for_all_models(models: List[ModelConfig]) -> dict | None:
    """
    Given an iterable of models (each with .id and .max_model_len),
    find the single best GPU that can host *all* of them at their
    max_model_len simultaneously.

    or None if no GPU can host all models.
    """

    models = list(models)
    num_models = len(models)

    gpu_total_vram: Dict[GPUS, float] = {}  # sum of VRAM across all models
    gpu_model_counts: Dict[GPUS, int] = (
        {}
    )  # how many models this GPU has been evaluated for

    for model in models:
        profile = get_model_vram_profile(model.id) or {}
        results = load_model_index_and_config(model.id)
        config = results.get("config")  # model config
        model_index = results.get("index")  # model index

        if not config:
            raise ValueError(f"Huggingface config not found for {model.id}")

        if model_index:
            total_size: int = model_index["metadata"]["total_size"]
        else:
            total_size = get_single_model_size(model.id)

        max_position_embeddings = config.get("max_position_embeddings")

        if model.max_model_len > max_position_embeddings:
            raise ValueError(
                f"{model.id} max_model_length exceeds model's max context of: {max_position_embeddings}"
            )

        for gpu in VRAM_BY_GPU.keys():
            if gpu in profile:
                details = profile[gpu]
                allocated_gb = details["allocated_gb"]  # baseline, no KV
                kv_overhead = details["kv_overhead"]  # KV for total_sequence_len
                total_sequence_len = details["total_sequence_len"]

                if total_sequence_len <= 0:
                    total_vram_for_max_len = approx_total_for_model
                else:
                    kv_per_token = kv_overhead / total_sequence_len
                    kv_for_max_len = (
                        kv_per_token * model.max_model_len
                    ) * model.max_model_concurrency
                    total_vram_for_max_len = allocated_gb + kv_for_max_len
            else:
                # Fallback for GPUs without profile
                num_kv_heads = config.get("num_key_value_heads")
                num_hidden_layers = config.get("num_hidden_layers")
                head_dim = config.get("head_dim")
                torch_dtype = config.get("torch_dtype")

                if any(
                    v is None
                    for v in (num_kv_heads, num_hidden_layers, head_dim, torch_dtype)
                ):
                    raise RuntimeError(
                        "One or more required config values are missing for vram calculation"
                    )

                dtype = map_hf_dtype(torch_dtype)

                model_vram_gb = model_vram_gib_from_total_size(total_size)

                kv_vram_bytes = kv_cache_bytes(
                    model.max_model_concurrency,
                    model.max_model_len,
                    num_hidden_layers,
                    num_kv_heads,
                    head_dim,
                    dtype,
                )
                kv_vram_gb = kv_vram_bytes / (1024**3)

                approx_total_for_model = model_vram_gb + kv_vram_gb
                total_vram_for_max_len = approx_total_for_model

            # Accumulate VRAM across models
            gpu_total_vram[gpu] = gpu_total_vram.get(gpu, 0.0) + total_vram_for_max_len
            gpu_model_counts[gpu] = gpu_model_counts.get(gpu, 0) + 1

    # Now pick the best GPU
    best_gpu = None
    best_capacity = None
    best_required_vram = None

    for gpu, total_required in gpu_total_vram.items():
        # Must be evaluated for all models
        if gpu_model_counts.get(gpu, 0) != num_models:
            continue

        try:
            capacity = get_vram_for_gpu(gpu)
        except ValueError:
            continue

        # Skip GPUs that can't actually fit all models
        if total_required > capacity:
            continue

        # Choose the smallest capacity GPU that fits.
        if (
            best_capacity is None
            or capacity < best_capacity
            or (capacity == best_capacity and total_required < best_required_vram)
        ):
            best_gpu = gpu
            best_capacity = capacity
            best_required_vram = total_required

    if best_gpu is None:
        return None  # No GPU can host all models

    return {
        "gpu": best_gpu,
        "required_vram_gb": best_required_vram,
        "gpu_capacity_gb": best_capacity,
        "headroom_gb": best_capacity - best_required_vram,
    }


def can_gpu_host_models(gpu: GPUS, models: List[ModelConfig]) -> dict | None:
    """
    Check whether a specific GPU can host all given models simultaneously

    Uses measured VRAM profiles when available, and falls back
    to an approximate estimate (base model + KV cache) when they are not.

    Returns a dict like:
        {
            "gpu": GPUS.XXX,
            "required_vram_gb": float,
            "gpu_capacity_gb": int,
            "headroom_gb": float,
        }

    or None if:
        - the GPU lacks a VRAM entry, or
        - total required VRAM exceeds the GPU's capacity.
    """
    models = list(models)
    total_required_vram = 0.0

    # Ensure the GPU is in our VRAM map
    try:
        gpu_capacity = get_vram_for_gpu(gpu)
    except ValueError:
        # GPU not registered in VRAM_BY_GPU
        return None

    for model in models:
        # Get profile (may be empty / None)
        profile = get_model_vram_profile(model.id) or {}

        # Load config & index (same pattern as best_gpu_for_all_models)
        results = load_model_index_and_config(model.id)
        config = results.get("config")  # model config
        model_index = results.get("index")  # model index

        if model_index:
            total_size: int = model_index["metadata"]["total_size"]
        else:
            total_size = get_single_model_size(model.id)

        max_position_embeddings = config.get("max_position_embeddings")

        if model.max_model_len > max_position_embeddings:
            raise ValueError(
                f"{model.id} max_model_length exceeds model's max context of: "
                f"{max_position_embeddings}"
            )

        # Use profile for this GPU if available, otherwise fallback
        if gpu in profile:
            details = profile[gpu]
            allocated_gb = details["allocated_gb"]  # baseline, no KV
            kv_overhead = details["kv_overhead"]  # KV for total_sequence_len
            total_sequence_len = details["total_sequence_len"]

            if total_sequence_len <= 0:
                # Bad profile entry; fallback to approximatfrom src.covenantlabs_sdk.client.types import e
                total_vram_for_max_len = approx_total_for_model
            else:
                kv_per_token = kv_overhead / total_sequence_len
                kv_for_max_len = (
                    kv_per_token * model.max_model_len
                ) * model.max_model_concurrency
                total_vram_for_max_len = allocated_gb + kv_for_max_len
        else:
            # No profile for this GPU/model combo – use approximate estimate]
            num_kv_heads = config.get("num_key_value_heads")
            num_hidden_layers = config.get("num_hidden_layers")
            head_dim = config.get("head_dim")
            torch_dtype = config.get("torch_dtype")
            dtype = map_hf_dtype(torch_dtype)

            model_vram_gb = model_vram_gib_from_total_size(total_size)

            kv_vram_bytes = kv_cache_bytes(
                model.max_model_concurrency,
                model.max_model_len,
                num_hidden_layers,
                num_kv_heads,
                head_dim,
                dtype,
            )
            kv_vram_gb = kv_vram_bytes / (1024**3)

            approx_total_for_model = model_vram_gb + kv_vram_gb
            total_vram_for_max_len = approx_total_for_model

        total_required_vram += total_vram_for_max_len

    # Finally check if everything fits on this GPU
    if total_required_vram > gpu_capacity:
        return None

    return {
        "gpu": gpu,
        "required_vram_gb": total_required_vram,
        "gpu_capacity_gb": gpu_capacity,
        "headroom_gb": gpu_capacity - total_required_vram,
    }


def kv_cache_bytes(
    batch: int,
    seq_len: int,
    num_hidden_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: DType,
) -> float:
    bpe = bytes_per_dtype(kv_dtype)
    return batch * seq_len * num_hidden_layers * num_kv_heads * head_dim * 2 * bpe


def kv_cache_gib(*args, **kwargs) -> float:
    return kv_cache_bytes(*args, **kwargs) / (1024**3)


def model_vram_gib_from_total_size(total_size_bytes: int) -> float:
    return total_size_bytes / (1024**3)


def model_params_gib(num_params: int, param_dtype: DType) -> float:
    """
    Compute VRAM (in GiB) used by model parameters.

    Parameters:
        num_params (int): Number of model parameters.
        param_dtype (DType): Data type of the parameters.

    Returns:
        float: VRAM in GiB used by the parameters.
    """
    bytes_per_param = bytes_per_dtype(param_dtype)
    total_bytes = num_params * bytes_per_param
    return total_bytes / (1024**3)


def dataclass_to_dict(obj: Any) -> dict:
    """
    Recursively convert a dataclass (or nested dataclass) into
    a nested dict/list/primitive structure.
    """
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = dataclass_to_dict(value)  # recurse
        return result
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj
