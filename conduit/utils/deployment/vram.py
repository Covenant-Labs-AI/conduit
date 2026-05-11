import bisect
import json
import os
import re
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin
from dataclasses import MISSING

import psutil
import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from conduit.conduit_types import LmLiteModelConfig
from conduit.utils import bytes_to_gib
from conduit.utils.accelerators.nvidia import architectures_for_dtype
from conduit.utils.model import DType, bytes_per_dtype, map_hf_dtype


from .models import LLMVramProfile

TOut = TypeVar("TOut")

VLLM_OVERHEAD_PER_GPU_GB: float = 15.0


# ----------------------------
# Hugging Face model metadata
# ----------------------------
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
        model_index = results.get("index")  # model index

        if model_index:
            size: int = model_index["metadata"]["total_size"]
        else:
            size = get_single_model_size(model.id)
        total_size += size

    return bytes_to_gib(total_size) + runtime_size_gb


# ----------------------------
# VRAM math utilities
# ----------------------------
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
    """
    bytes_per_param = bytes_per_dtype(param_dtype)
    total_bytes = num_params * bytes_per_param
    return total_bytes / (1024**3)


def has_enough_disk_space(required_bytes: int) -> bool:
    path = os.getcwd()
    du = psutil.disk_usage(path)
    return du.free >= required_bytes


# ----------------------------
# TP / rank helper utilities
# ----------------------------
def _divisors(n: int) -> list[int]:
    return [i for i in range(1, n + 1) if n % i == 0]


def _common_elements(*lists: list[int]) -> list[int]:
    if not lists:
        return []
    s = set(lists[0])
    for lst in lists[1:]:
        s &= set(lst)
    return sorted(s)


def _next_largest(sorted_list: list[int], x: int) -> int | None:
    idx = bisect.bisect_right(sorted_list, x)
    return sorted_list[idx] if idx < len(sorted_list) else None


def _valid_tp_sizes_from_config(model_profile: LLMVramProfile) -> list[int]:
    attention_heads = model_profile.num_attention_heads
    hidden_size = model_profile.hidden_size

    if not isinstance(attention_heads, int) or not isinstance(hidden_size, int):
        return [1]

    attention_divs = _divisors(attention_heads)
    hidden_divs = _divisors(hidden_size)

    valid = _common_elements(attention_divs, hidden_divs)

    # Ensure 1 is always present
    if 1 not in valid:
        valid = [1] + valid
        valid = sorted(set(valid))
    return valid


def _infer_dtype_string(config, *, raise_on_fail=True, default=None):
    """
    Returns dtype as a STRING like: 'float16', 'bfloat16', 'float32', 'float64'
    Looks in common keys; falls back to regex on stringified config; else raise/default.
    """

    def get(k):
        return config.get(k) if isinstance(config, dict) else getattr(config, k, None)

    # 1) direct keys in priority order
    for k in ("torch_dtype", "dtype", "compute_dtype", "weights_dtype", "param_dtype"):
        v = get(k)
        if v is None:
            continue

        # torch dtype objects stringify like 'torch.float16'
        s = str(v).strip().lower()
        s = s.replace("torch.", "")

        # normalize common aliases
        alias = {
            "fp16": "float16",
            "half": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
            "fp64": "float64",
        }
        s = alias.get(s, s)

        if s in ("float16", "bfloat16", "float32", "float64"):
            return s

    # 2) flags sometimes used instead of dtype
    if get("bf16") is True:
        return "bfloat16"
    if get("fp16") is True:
        return "float16"

    # 3) regex fallback on the whole config text
    text = str(config).lower()
    m = re.search(
        r"\b(torch\.)?(bfloat16|bf16|float16|fp16|float32|fp32|float64|fp64)\b", text
    )
    if m:
        s = m.group(2)
        return {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
            "fp64": "float64",
        }.get(s, s)

    # 4) fail
    if raise_on_fail:
        raise KeyError("No dtype-like key/value found in config")
    return default


# ----------------------------
# LLM profile building
# ----------------------------


def try_build_llm_vram_profile_local(model: Any) -> LLMVramProfile:
    results: Dict[str, Any] = load_model_index_and_config(model.id)
    config: Dict[str, Any] = results.get("config") or {}

    text_config = config.get("text_config") or {}

    max_position_embeddings = text_config.get(
        "max_position_embeddings", config.get("max_position_embeddings")
    )
    print(max_position_embeddings)
    requested_max_len = int(getattr(model, "max_model_len"))
    if max_position_embeddings is not None and requested_max_len > int(
        max_position_embeddings
    ):
        raise ValueError(
            f"Requested context length ({requested_max_len}) is larger than "
            f"model's max_position_embeddings ({max_position_embeddings}); cannot host this model."
        )

    dtype_str = _infer_dtype_string(text_config or config)

    quant_config = config.get("quantization_config") or None
    qdtype_str = None

    if quant_config:
        qdtype_str = (
            quant_config.get("compute_dtype")
            or quant_config.get("quant_method")
            or quant_config.get("bnb_4bit_compute_dtype")
            or quant_config.get("torch_dtype")
        )

    num_hidden_layers = text_config.get(
        "num_hidden_layers", config.get("num_hidden_layers")
    )
    if num_hidden_layers is None:
        raise ValueError("Config missing num_hidden_layers")

    num_attention_heads = text_config.get(
        "num_attention_heads", config.get("num_attention_heads")
    )

    num_kv_heads = text_config.get(
        "num_key_value_heads", config.get("num_key_value_heads")
    )
    if num_kv_heads is None:
        if num_attention_heads is None:
            raise ValueError(
                "Config missing both num_key_value_heads and num_attention_heads"
            )
        num_kv_heads = num_attention_heads

    head_dim = text_config.get("head_dim", config.get("head_dim"))
    hidden_size = text_config.get("hidden_size", config.get("hidden_size"))
    if head_dim is None:
        if hidden_size is None or num_attention_heads is None:
            raise ValueError(
                "Cannot derive head_dim: need head_dim or (hidden_size and num_attention_heads)"
            )
        head_dim = int(hidden_size) // int(num_attention_heads)

    return LLMVramProfile(
        max_position_embeddings=max_position_embeddings,
        dtype=dtype_str,
        quant_dtype=qdtype_str,
        num_hidden_layers=int(num_hidden_layers),
        num_attention_heads=(
            int(num_attention_heads) if num_attention_heads is not None else None
        ),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        hidden_size=int(hidden_size) if hidden_size is not None else None,
    )


def call_llm_vram_profile_agent(model_id: str) -> LLMVramProfile:
    revision = "main"
    config_url = f"https://huggingface.co/{model_id}/resolve/{revision}/config.json"

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    hf_headers = {"Authorization": f"Bearer {token}"} if token else {}

    cfg_text = requests.get(config_url, headers=hf_headers, timeout=30).text

    base_url = os.getenv("COVENANT_HF_AGENT_PARSER_BASE_URL")
    parser_url = f"{base_url}/v1/profile/from_hf_config"

    resp = requests.post(
        parser_url,
        json={"raw_json": cfg_text},
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()

    return LLMVramProfile(**resp.json())


def parse_llm_json(json_response: str, output_type: Type[TOut]) -> TOut:
    """
    Safely deserialize an LLM JSON response into the provided dataclass type.

    Smart recovery behavior:
    - strips content before </think>
    - attempts to extract the first JSON object/array if extra text is present
    - if the output dataclass has exactly one field and the model returns a bare
      top-level value compatible with that field, wraps it automatically

    Example:
        @dataclass
        class ActionPolicyResult:
            desired_actions: List[Action]

        Bare output:
            [{...}, {...}]

        Repaired to:
            {"desired_actions": [{...}, {...}]}
    """

    def _is_optional_type(target_type: Any) -> bool:
        origin = get_origin(target_type)
        if origin is Union:
            return any(t is type(None) for t in get_args(target_type))
        return False

    def _has_default(field_obj) -> bool:
        return (
            field_obj.default is not MISSING or field_obj.default_factory is not MISSING
        )

    def _extract_json_payload(text: str) -> str:
        text = text.strip()

        think_end = "</think>"
        if think_end in text:
            text = text.split(think_end, 1)[1].strip()

        # Fast path
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try to extract from first { or [
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        if not starts:
            return text

        candidate = text[min(starts) :].strip()
        return candidate

    def _coerce(value: Any, target_type: Any) -> Any:
        if value is None:
            if _is_optional_type(target_type):
                return None
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

            kwargs = {}
            missing_required = []

            for f in fields(target_type):
                if f.name in value:
                    kwargs[f.name] = _coerce(value[f.name], f.type)
                else:
                    if _has_default(f):
                        continue
                    if _is_optional_type(f.type):
                        kwargs[f.name] = None
                    else:
                        missing_required.append(f.name)

            if missing_required:
                raise ValueError(
                    f"Missing required fields for {target_type.__name__}: "
                    + ", ".join(sorted(missing_required))
                )

            return target_type(**kwargs)

        return value

    def _repair_top_level_shape(data: Any, target_type: Type[TOut]) -> Any:
        """
        Safe repair rules only.

        If the target is a dataclass with exactly one field, and the model returned a
        bare value of the right general shape for that field, wrap it automatically.

        Examples:
            [{"x": 1}] -> {"items": [{"x": 1}]}
            "hello"    -> {"result": "hello"}
        """
        if not is_dataclass(target_type):
            return data

        if isinstance(data, dict):
            return data

        top_fields = list(fields(target_type))
        if len(top_fields) != 1:
            return data

        only_field = top_fields[0]
        field_type = only_field.type
        origin = get_origin(field_type)

        # Safe wrapper for list field
        if origin in (list, List) and isinstance(data, list):
            return {only_field.name: data}

        # Optional[List[T]] style wrapper
        if origin is Union:
            non_none = [t for t in get_args(field_type) if t is not type(None)]
            if len(non_none) == 1:
                inner = non_none[0]
                inner_origin = get_origin(inner)
                if inner_origin in (list, List) and isinstance(data, list):
                    return {only_field.name: data}

        return data

    if not is_dataclass(output_type):
        raise TypeError("output_type must be a dataclass type")

    json_response = _extract_json_payload(json_response)

    try:
        data = json.loads(json_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON from LLM: {e.msg}\n\nFull response:\n{json_response}"
        ) from e

    repaired_data = _repair_top_level_shape(data, output_type)

    if not isinstance(repaired_data, dict):
        top_fields = list(fields(output_type))
        if len(top_fields) == 1:
            expected = f'{{"{top_fields[0].name}": ...}}'
        else:
            expected = "top-level JSON object"
        raise ValueError(
            f"Expected top-level JSON object for {output_type.__name__}, "
            f"got {type(repaired_data).__name__}. Expected shape like {expected}"
        )

    output_fields = {f.name for f in fields(output_type)}
    missing = []
    for f in fields(output_type):
        if (
            f.name not in repaired_data
            and not _has_default(f)
            and not _is_optional_type(f.type)
        ):
            missing.append(f.name)

    if missing:
        raise ValueError(
            f"Missing expected fields in LLM response: {', '.join(sorted(missing))}"
        )

    try:
        return _coerce(repaired_data, output_type)
    except Exception as e:
        raise ValueError(
            f"Failed to deserialize LLM JSON into {output_type.__name__}: {e}"
        ) from e
