import os
import json
import math
import psutil
import hashlib
from enum import Enum
from dataclasses import is_dataclass, asdict, fields, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    List,
    Type,
    TypeVar,
    Dict,
    get_args,
    get_origin,
    Union,
)
from conduit.utils import ComputeOffering, bytes_to_gib, gib_to_mib, mib_to_gib
from conduit.utils.accelerators.nvidia import (
    GPUHostingResult,
    NvidiaGPU,
    architectures_for_dtype,
)
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import HfHubHTTPError
from conduit.conduit_types import (
    LmLiteModelConfig,
)
from conduit.utils.model import bytes_per_dtype, DType, map_hf_dtype

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

if TYPE_CHECKING:
    from conduit.runtime import LMLiteBlock


class DeploymentConstraint(Enum):
    ENTERPRISE = "ENTERPRISE"
    SINGLE_DEVICE = "SINGLE_DEVICE"


@dataclass
class ComputeOfferingCandidate:
    offering: ComputeOffering
    gpu: NvidiaGPU
    num_gpu: int
    price: int


def calculate_best_compute_offering(
    compute_offerings: List[ComputeOffering],
    nvidia_gpus: List[NvidiaGPU],
    models: List[LmLiteModelConfig],
    constraints: List[DeploymentConstraint],
) -> GPUHostingResult:
    total_param_vram_gb = 0.0
    total_kv_vram_gb = 0.0
    for model in models:
        results = load_model_index_and_config(model.id)
        config = results.get("config") or {}
        model_index = results.get("index")

        if model_index:
            total_size: int = model_index["metadata"]["total_size"]
        else:
            total_size = get_single_model_size(model.id)

        max_position_embeddings = config.get("max_position_embeddings")
        if (
            max_position_embeddings is not None
            and model.max_model_len > max_position_embeddings
        ):
            # Cannot host this model at requested context length
            raise ValueError(
                f"Requested context length ({model.max_model_len}) is larger than "
                f"model's max_position_embeddings ({max_position_embeddings}); "
                f"cannot host this model."
            )

        torch_dtype = config.get("torch_dtype")
        dtype = map_hf_dtype(torch_dtype)
        quant_config = config.get("quantization_config")

        if quant_config:
            quant_dtype = quant_config.get("quant_method")
            quant_dtype = map_hf_dtype(quant_dtype)
            supported_arches = architectures_for_dtype(quant_dtype)
        else:
            supported_arches = architectures_for_dtype(dtype)

        num_kv_heads = config.get("num_key_value_heads")
        num_hidden_layers = config.get("num_hidden_layers")
        head_dim = config.get("head_dim")

        # ---- PARAMS (weights) VRAM ----
        model_vram_gb = model_vram_gib_from_total_size(total_size)

        # ---- KV CACHE VRAM ----
        kv_vram_bytes = kv_cache_bytes(
            model.max_model_concurrency,
            model.max_model_len,
            num_hidden_layers,
            num_kv_heads,
            head_dim,
            dtype,  # kv most likey in the same dtype as model, not quantized
        )
        kv_vram_gb = kv_vram_bytes / (1024**3)

        total_param_vram_gb += model_vram_gb
        total_kv_vram_gb += kv_vram_gb

    raw_total_vram_gb = total_param_vram_gb + total_kv_vram_gb

    required_vram_gb = raw_total_vram_gb

    single_device = DeploymentConstraint.SINGLE_DEVICE in constraints

    candidates = []
    for offering in compute_offerings:
        gpu = next((u for u in nvidia_gpus if u.name == offering.id), None)

        if (
            not offering.enterprise_grade
            and DeploymentConstraint.ENTERPRISE in constraints
        ):
            continue

        if not gpu:
            continue

        if gpu.architecture not in supported_arches:
            continue

        chip_memory = gpu.memory_free_mib
        n_chips = gib_to_mib(required_vram_gb) / chip_memory
        num_gpu = math.ceil(n_chips)

        if single_device and num_gpu > 1:
            continue

        if num_gpu > offering.max_available:
            continue

        gpu_prices = math.ceil(n_chips) * offering.price_per_hour
        candidate = ComputeOfferingCandidate(
            offering=offering, gpu=gpu, num_gpu=num_gpu, price=gpu_prices
        )
        candidates.append(candidate)

    candidates_sorted = sorted(candidates, key=lambda c: int(c.price))
    best_offering: ComputeOffering = candidates_sorted[0].offering
    best_offering_gpu_count = candidates_sorted[0].num_gpu
    best_gpu: NvidiaGPU = candidates_sorted[0].gpu
    headroom_gb = mib_to_gib(best_gpu.memory_mib) - required_vram_gb
    best_gpu.gpu_count = best_offering_gpu_count

    return GPUHostingResult(
        gpus=best_gpu,
        compute_offering=best_offering,
        required_vram_gb=required_vram_gb,
        raw_model_vram_gb=raw_total_vram_gb,
        total_capacity_gb=mib_to_gib(best_gpu.memory_mib),
        headroom_gb=headroom_gb,
        price_cents=best_offering.price_per_hour,
        can_host=True,
    )


def can_nvidia_gpus_host_models(
    gpus: NvidiaGPU,
    models: List[LmLiteModelConfig],
) -> GPUHostingResult:
    """
    Non-raising variant of `can_nvidia_gpus_host_models_or_raise`.

    Always returns a GPUHostingResult. If the inputs are invalid or the models
    cannot be hosted, `can_host=False` and `error` is populated.

    Notes:
    - Params (weights) + KV cache are estimated in GiB.
    - KV cache is required to fit entirely in GPU VRAM.
    - No CPU offloading is supported.
    """

    models = list(models)

    def _fail(msg: str) -> GPUHostingResult:
        total_gpu_free_capacity_gb = (gpus.memory_free_mib * gpus.gpu_count) / 1024.0
        return GPUHostingResult(
            gpus=gpus,
            price_cents=0,
            required_vram_gb=0.0,
            raw_model_vram_gb=0.0,
            total_capacity_gb=total_gpu_free_capacity_gb,
            headroom_gb=total_gpu_free_capacity_gb,
            can_host=False,
        )

    if not gpus:
        return _fail("can_nvidia_gpus_host_models requires at least one GPU")

    total_gpu_free_capacity_gb = (gpus.memory_free_mib * gpus.gpu_count) / 1024.0

    total_param_vram_gb = 0.0
    total_kv_vram_gb = 0.0

    try:
        for model in models:
            results = load_model_index_and_config(model.id)
            config = results.get("config") or {}
            model_index = results.get("index")

            if model_index:
                total_size: int = model_index["metadata"]["total_size"]
            else:
                total_size = get_single_model_size(model.id)

            max_position_embeddings = config.get("max_position_embeddings")
            if (
                max_position_embeddings is not None
                and model.max_model_len > max_position_embeddings
            ):
                return _fail(
                    f"Requested context length ({model.max_model_len}) is larger than "
                    f"model's max_position_embeddings ({max_position_embeddings}); "
                    f"cannot host this model."
                )

            torch_dtype = config.get("torch_dtype")
            dtype = map_hf_dtype(torch_dtype)
            quant_config = config.get("quantization_config")

            if quant_config:
                quant_dtype = quant_config.get("quant_method")
                quant_dtype = map_hf_dtype(quant_dtype)
                supported_arches = architectures_for_dtype(quant_dtype)
            else:
                supported_arches = architectures_for_dtype(dtype)

            if not gpus.architecture in supported_arches:
                return _fail(
                    f"{gpus.architecture} not supported "
                    f"supported architectures: {supported_arches}"
                )

            num_kv_heads = config.get("num_key_value_heads")
            num_hidden_layers = config.get("num_hidden_layers")
            head_dim = config.get("head_dim")

            # ---- PARAMS (weights) VRAM ----
            model_vram_gb = model_vram_gib_from_total_size(total_size)

            # ---- KV CACHE VRAM ----
            kv_vram_bytes = kv_cache_bytes(
                model.max_model_concurrency,
                model.max_model_len,
                num_hidden_layers,
                num_kv_heads,
                head_dim,
                dtype,  # kv most likely in the same dtype as model, not quantized
            )
            kv_vram_gb = kv_vram_bytes / (1024**3)

            total_param_vram_gb += model_vram_gb
            total_kv_vram_gb += kv_vram_gb

    except Exception as e:
        # Catch unexpected config/index/dtype parsing issues and return them as a failure.
        return _fail(str(e))

    raw_total_vram_gb = total_param_vram_gb + total_kv_vram_gb
    required_vram_gb = raw_total_vram_gb
    total_capacity_gb = total_gpu_free_capacity_gb

    can_host = required_vram_gb <= total_gpu_free_capacity_gb
    headroom_gb = total_gpu_free_capacity_gb - required_vram_gb

    return GPUHostingResult(
        gpus=gpus,
        price_cents=0,
        required_vram_gb=required_vram_gb,
        raw_model_vram_gb=raw_total_vram_gb,
        total_capacity_gb=total_capacity_gb,
        headroom_gb=headroom_gb,
        can_host=can_host,
    )


def can_nvidia_gpus_host_models_or_raise(
    gpus: List[NvidiaGPU],
    models: List[LmLiteModelConfig],
) -> GPUHostingResult:
    """
    Check whether a set of Nvidia GPUs can host all given models simultaneously.

    - Params (weights) + KV cache are estimated in GiB.
    - KV cache is required to fit entirely in GPU VRAM.
    - No CPU offloading is supported. If the combined params + KV cache +
      packing overhead exceed total free GPU VRAM, a ValueError is raised.
    """

    gpus = list(gpus)
    models = list(models)

    if not gpus:
        raise ValueError("can_nvidia_gpus_host_models requires at least one GPU")

    total_gpu_free_capacity_gb = sum(gpu.memory_free_mib for gpu in gpus) / 1024.0

    total_param_vram_gb = 0.0
    total_kv_vram_gb = 0.0
    for model in models:
        results = load_model_index_and_config(model.id)
        config = results.get("config") or {}
        model_index = results.get("index")

        if model_index:
            total_size: int = model_index["metadata"]["total_size"]
        else:
            total_size = get_single_model_size(model.id)

        max_position_embeddings = config.get("max_position_embeddings")
        if (
            max_position_embeddings is not None
            and model.max_model_len > max_position_embeddings
        ):
            # Cannot host this model at requested context length
            raise ValueError(
                f"Requested context length ({model.max_model_len}) is larger than "
                f"model's max_position_embeddings ({max_position_embeddings}); "
                f"cannot host this model."
            )

        torch_dtype = config.get("torch_dtype")
        dtype = map_hf_dtype(torch_dtype)
        quant_config = config.get("quantization_config")

        if quant_config:
            quant_dtype = quant_config.get("quant_method")
            quant_dtype = map_hf_dtype(quant_dtype)
            supported_arches = architectures_for_dtype(quant_dtype)
        else:
            supported_arches = architectures_for_dtype(dtype)

        if not any(gpu.architecture in supported_arches for gpu in gpus):
            raise ValueError(
                f"None of the GPUs ({[gpu.architecture for gpu in gpus]}) are in the list of "
                f"supported architectures: {supported_arches}"
            )

        num_kv_heads = config.get("num_key_value_heads")
        num_hidden_layers = config.get("num_hidden_layers")
        head_dim = config.get("head_dim")

        # ---- PARAMS (weights) VRAM ----
        model_vram_gb = model_vram_gib_from_total_size(total_size)

        # ---- KV CACHE VRAM ----
        kv_vram_bytes = kv_cache_bytes(
            model.max_model_concurrency,
            model.max_model_len,
            num_hidden_layers,
            num_kv_heads,
            head_dim,
            dtype,  # kv most likey in the same dtype as model, not quantized
        )
        kv_vram_gb = kv_vram_bytes / (1024**3)

        total_param_vram_gb += model_vram_gb
        total_kv_vram_gb += kv_vram_gb

    raw_total_vram_gb = total_param_vram_gb + total_kv_vram_gb

    required_vram_gb = raw_total_vram_gb
    total_capacity_gb = total_gpu_free_capacity_gb  # GPU only

    if required_vram_gb > total_gpu_free_capacity_gb:
        # Model params + KV cache + packing overhead don't fit on GPU
        raise ValueError(
            "Cannot host models: required VRAM for parameters + KV cache "
            "overhead ({req:.2f} GiB) exceeds total free GPU VRAM ({gpu:.2f} GiB).".format(
                req=required_vram_gb,
                gpu=total_gpu_free_capacity_gb,
            )
        )

    headroom_gb = total_gpu_free_capacity_gb - required_vram_gb

    return GPUHostingResult(
        gpus=gpus,
        required_vram_gb=required_vram_gb,
        price_cents=0,
        raw_model_vram_gb=raw_total_vram_gb,
        total_capacity_gb=total_capacity_gb,
        headroom_gb=headroom_gb,
        can_host=True,
    )


def gpu_id_and_count_label(gpus: "NvidiaGPU", sep: str = ", ") -> tuple[str, str, int]:
    if not gpus:
        return ("", "", 0)

    gpu_id = gpus.name
    num_gpus = gpus.gpu_count
    display = f"{gpu_id}x {num_gpus}"
    return (gpu_id, display, num_gpus)


def compute_deployment_key(block: "LMLiteBlock") -> str:
    payload = {
        "runtime": block.runtime.value,
        "image": block.image,
        "provider": block.compute_provider.value,
        "constraints": block.constraints,
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
    raw = json.dumps(
        payload,
        sort_keys=True,
        default=lambda o: o.value if isinstance(o, Enum) else str(o),
    )
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


def calculate_container_size_gb(
    models: List[LmLiteModelConfig], runtime_size_gb=0
) -> int:
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

    return bytes_to_gib(total_size) + runtime_size_gb


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


def has_enough_disk_space(required_bytes: int) -> bool:
    path = os.getcwd()
    du = psutil.disk_usage(path)
    return du.free >= required_bytes


def parse_llm_json(json_response: str, output_type: Type[TOut]) -> TOut:
    """
    Safely deserialize an LLM JSON response into the provided dataclass type.
    Supports nested dataclasses and lists of dataclasses.
    """

    def _coerce(value: Any, target_type: Any) -> Any:
        """
        Recursively convert `value` (from JSON) into `target_type`:
        - dataclasses
        - List[T]
        - Dict[K,V] (basic)
        - Optional/Union
        - primitives
        """
        if value is None:
            return None

        origin = get_origin(target_type)
        args = get_args(target_type)

        if origin is Union:
            non_none = [t for t in args if t is not type(None)]
            if len(non_none) == 1:
                return _coerce(value, non_none[0])
            last_err = None
            for t in non_none:
                try:
                    return _coerce(value, t)
                except Exception as e:
                    last_err = e
            raise ValueError(
                f"Could not coerce {value!r} into {target_type}: {last_err}"
            ) from last_err

        if origin in (list, List):
            (elem_type,) = args or (Any,)
            if not isinstance(value, list):
                raise ValueError(
                    f"Expected list for {target_type}, got {type(value).__name__}"
                )
            return [_coerce(v, elem_type) for v in value]

        if origin in (dict, Dict):
            key_type, val_type = args or (Any, Any)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected dict for {target_type}, got {type(value).__name__}"
                )
            return {
                _coerce(k, key_type): _coerce(v, val_type) for k, v in value.items()
            }

        if is_dataclass(target_type):
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected object (dict) for {target_type.__name__}, got {type(value).__name__}"
                )

            fdefs = {f.name: f for f in fields(target_type)}
            missing = {name for name, f in fdefs.items() if name not in value and f.default is f.default_factory}  # type: ignore
            kwargs = {}
            for name, f in fdefs.items():
                if name in value:
                    kwargs[name] = _coerce(value[name], f.type)

            required = {
                f.name
                for f in fields(target_type)
                if f.default is f.default_factory
                and f.default_factory is f.default_factory
            }  # type: ignore

            return target_type(**kwargs)

        return value

    if not is_dataclass(output_type):
        raise TypeError("output_type must be a dataclass type")

    think_end = "</think>"
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
        return _coerce(data, output_type)
    except Exception as e:
        raise ValueError(
            f"Failed to deserialize LLM JSON into {output_type.__name__}: {e}"
        ) from e
