from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Type, TypeVar, overload

from conduit.blocks import OpenAICompatableRuntimeBlock
from conduit.conduit_http import OpenAIMessage
from conduit.conduit_types import ComputeProvider, LmLiteModelConfig, VLLMModelConfig
from conduit.runtime import LMLiteBlock, VLLMBlock
from conduit.utils.deployment import DeploymentConstraint

TOut = TypeVar("TOut")


# -----------------------------
# lightweight transform handle
# -----------------------------
class Transform:
    def __init__(self, manager: "TransformManager", id: str):
        self.manager = manager
        self.id = id

    @property
    def ready(self) -> bool:
        return self.manager.transform_ready(self.id)

    @overload
    def __call__(
        self,
        model_id: str | None,
        messages: List[OpenAIMessage],
        guidance: str | None = None,
        *,
        output: None = ...,
        input: None = ...,
    ) -> str: ...

    @overload
    def __call__(
        self,
        model_id: str | None = ...,
        messages: None = ...,
        guidance: str | None = None,
        *,
        input: Any,
        output: Type[TOut],
    ) -> TOut: ...

    def __call__(
        self,
        model_id: str | None = None,
        messages: List[OpenAIMessage] | None = None,
        guidance: str | None = None,
        *,
        input: Any = None,
        output: Type[TOut] | None = None,
    ) -> Any:
        return self.manager.call_transform(
            self.id,
            model_id=model_id,
            messages=messages,
            guidance=guidance,
            input=input,
            output=output,
        )


# -----------------------------
# transform manager
# -----------------------------
class TransformManager:
    """
    Loads standardized transform spec, builds one runtime per compute_hash,
    and routes named transform calls to the correct runtime instance.

    Supported spec shape (official):
    {
      "transform_calls": {
        "lm_lite": {
          "compute_hash": "...",
          "compute_provider": "RUNPOD",
          "models": [...],
          "gpu": {"name": "NVIDIA L4", "gpu_count": 1},
          "replicas": 1,
          "constraints": [],
          "transforms": [
            {"id": "summarize_v1", "model": "Qwen/..."}
          ]
        }
      }
    }
    """

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec

        # normalized maps
        self.runtimes_by_hash: dict[str, dict[str, Any]] = {}
        self.transforms_by_id: dict[str, dict[str, Any]] = {}

        # instantiated runtime blocks (keyed by compute_hash)
        self.runtime_instances: dict[str, Any] = {}

        self._normalize_spec()
        self._build_runtime_pool()

    # ---------- public API ----------
    def transform(self, id: str) -> Transform:
        if id not in self.transforms_by_id:
            raise KeyError(f"Unknown transform id: {id}")
        return Transform(self, id)

    def transform_ready(self, id: str) -> bool:
        t = self._get_transform_spec(id)
        runtime = self._get_runtime_for_transform_spec(t)
        return bool(getattr(runtime, "ready", True))

    def call_transform(
        self,
        id: str,
        *,
        model_id: str | None = None,
        messages: List[OpenAIMessage] | None = None,
        guidance: str | None = None,
        input: Any = None,
        output: Type[TOut] | None = None,
    ) -> Any:
        t_spec = self._get_transform_spec(id)
        runtime = self._get_runtime_for_transform_spec(t_spec)

        # Official transform field is "model"
        resolved_model_id = model_id or t_spec.get("model")
        if not resolved_model_id:
            raise ValueError(
                f"Transform '{id}' requires model_id (none passed and no 'model' in spec)."
            )

        return runtime(
            model_id=resolved_model_id,
            messages=messages,
            guidance=guidance,
            input=input,
            output=output,
        )

    def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for compute_hash, runtime in self.runtime_instances.items():
            if hasattr(runtime, "health"):
                out[compute_hash] = runtime.health()
            else:
                out[compute_hash] = {"ready": bool(getattr(runtime, "ready", True))}
        return out

    def stop_all(self) -> None:
        for runtime in self.runtime_instances.values():
            if hasattr(runtime, "stop"):
                runtime.stop()

    def restart_all(self) -> None:
        for runtime in self.runtime_instances.values():
            if hasattr(runtime, "restart"):
                runtime.restart()

    def delete_all(self) -> None:
        for runtime in self.runtime_instances.values():
            if hasattr(runtime, "delete"):
                runtime.delete()

    # ---------- spec loading helpers ----------
    @classmethod
    def from_json_file(cls, path: str | Path) -> "TransformManager":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            spec = json.load(f)
        return cls(spec)

    # ---------- normalization ----------
    def _normalize_spec(self) -> None:
        """
        Supports:
        1) normalized shape:
           { "runtimes": {...}, "transforms": {...} }

           In this shape, transform entries may use either:
             - "model" (official)
             - "default_model_id" (legacy compatibility)

        2) grouped official shape:
           { "transform_calls": { "lm_lite": {...}, "vllm": {...}, ... } }
        """
        if "runtimes" in self.spec and "transforms" in self.spec:
            self.runtimes_by_hash = dict(self.spec["runtimes"])

            # normalize transforms to official "model" field
            raw_transforms = dict(self.spec["transforms"])
            normalized_transforms: dict[str, dict[str, Any]] = {}
            for tid, t in raw_transforms.items():
                if not isinstance(t, dict):
                    raise ValueError(f"Transform '{tid}' must be an object")
                normalized_transforms[tid] = {
                    **t,
                    "model": t.get("model") or t.get("default_model_id"),
                }
            self.transforms_by_id = normalized_transforms
            return

        grouped = self.spec.get("transform_calls")
        if not grouped:
            raise ValueError(
                "Spec must contain either ('runtimes' + 'transforms') or 'transform_calls'."
            )

        if isinstance(grouped, str):
            grouped = json.loads(grouped)

        if not isinstance(grouped, dict):
            raise ValueError("'transform_calls' must be a dict or JSON object string")

        runtimes_by_hash: dict[str, dict[str, Any]] = {}
        transforms_by_id: dict[str, dict[str, Any]] = {}

        # Expected grouped shape:
        # {
        #   "lm_lite": {
        #     "compute_hash": "...",
        #     "models": [...],
        #     "gpu": {"name": "...", "gpu_count": 1},
        #     "transforms": [{"id": "...", "model": "..."}]
        #   },
        #   "vllm": {...}
        # }
        for runtime_name, entry in grouped.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Runtime group '{runtime_name}' must be an object")

            compute_hash = entry.get("compute_hash")
            if not compute_hash:
                raise ValueError(
                    f"Missing compute_hash for runtime group '{runtime_name}'"
                )

            # Optional strict check for official gpu object shape
            gpu = entry.get("gpu")
            if isinstance(gpu, dict) and "gpu_count" not in gpu:
                raise ValueError(
                    f"Runtime group '{runtime_name}' gpu object must include 'gpu_count'"
                )

            # Store runtime spec keyed by compute_hash
            runtimes_by_hash[str(compute_hash)] = {
                "runtime": runtime_name,
                **{k: v for k, v in entry.items() if k != "transforms"},
            }

            # If no explicit transforms list, allow a single implicit transform keyed by compute_hash
            if "transforms" not in entry:
                transforms_by_id[str(compute_hash)] = {
                    "compute_hash": str(compute_hash),
                    "model": self._first_model_id(entry),
                }
                continue

            t_list = entry.get("transforms", [])
            if not isinstance(t_list, list):
                raise ValueError(
                    f"'transforms' for runtime group '{runtime_name}' must be a list"
                )

            for t in t_list:
                if not isinstance(t, dict):
                    raise ValueError(
                        f"Each transform in runtime group '{runtime_name}' must be an object"
                    )

                tid = t.get("id")
                if not tid:
                    raise ValueError(
                        f"Transform entry in group '{runtime_name}' missing 'id'"
                    )

                transform_model = t.get("model") or t.get("default_model_id")
                if not transform_model:
                    transform_model = self._first_model_id(entry)

                if not transform_model:
                    raise ValueError(
                        f"Transform '{tid}' in group '{runtime_name}' must include 'model' "
                        "or runtime group must define at least one model."
                    )

                transforms_by_id[str(tid)] = {
                    "compute_hash": str(compute_hash),
                    "model": transform_model,
                    **t,
                }

        self.runtimes_by_hash = runtimes_by_hash
        self.transforms_by_id = transforms_by_id

    def _first_model_id(self, runtime_entry: dict[str, Any]) -> str | None:
        models = runtime_entry.get("models") or []
        if isinstance(models, list) and models:
            m0 = models[0]
            if isinstance(m0, dict):
                return m0.get("model_id") or m0.get("id")
        return None

    # ---------- runtime pool ----------
    def _build_runtime_pool(self) -> None:
        for compute_hash, runtime_spec in self.runtimes_by_hash.items():
            self.runtime_instances[compute_hash] = self._build_runtime_instance(
                compute_hash=compute_hash,
                runtime_spec=runtime_spec,
            )

    def _build_runtime_instance(
        self, *, compute_hash: str, runtime_spec: dict[str, Any]
    ) -> Any:
        runtime_name = str(runtime_spec.get("runtime", "")).lower()

        gpu_name, gpu_count = self._parse_gpu(runtime_spec.get("gpu"))
        resolved_num_gpus = (
            gpu_count if gpu_count is not None else int(runtime_spec.get("num_gpus", 1))
        )

        if runtime_name == "lm_lite":
            return LMLiteBlock(
                models=self._parse_lmlite_models(runtime_spec),
                compute_provider=self._parse_compute_provider(
                    runtime_spec.get("compute_provider", "LOCAL")
                ),
                gpu=gpu_name,
                constraints=self._parse_constraints(
                    runtime_spec.get("constraints", [])
                ),
                replicas=int(runtime_spec.get("replicas", 1)),
                num_gpus=resolved_num_gpus,
                compute_provider_config_overrides=runtime_spec.get(
                    "compute_provider_config_overrides"
                ),
            )

        if runtime_name == "vllm":
            return VLLMBlock(
                models=self._parse_vllm_models(runtime_spec),
                compute_provider=self._parse_compute_provider(
                    runtime_spec.get("compute_provider", "LOCAL")
                ),
                gpu=gpu_name,
                constraints=self._parse_constraints(
                    runtime_spec.get("constraints", [])
                ),
                replicas=int(runtime_spec.get("replicas", 1)),
                num_gpus=resolved_num_gpus,
                compute_provider_config_overrides=runtime_spec.get(
                    "compute_provider_config_overrides"
                ),
            )

        if runtime_name == "openai_compat":
            # Stateless/non-provisioned runtime; compute_hash still works as logical grouping
            return OpenAICompatableRuntimeBlock(
                host=runtime_spec.get("host", "api.openai.com"),
                port=runtime_spec.get("port", 443),
                scheme=runtime_spec.get("scheme", "https"),
                api_key_env=runtime_spec.get("api_key_env", "OPENAI_API_KEY"),
                require_api_key=runtime_spec.get("require_api_key", True),
            )

        raise ValueError(
            f"Unsupported runtime '{runtime_name}' for compute_hash={compute_hash}"
        )

    # ---------- parsing ----------
    def _parse_compute_provider(self, value: str) -> ComputeProvider:
        try:
            return ComputeProvider[str(value).upper()]
        except KeyError as e:
            raise ValueError(f"Invalid compute_provider: {value}") from e

    def _parse_constraints(self, values: list[str]) -> list[DeploymentConstraint]:
        out: list[DeploymentConstraint] = []
        for v in values or []:
            key = str(v).upper()
            try:
                out.append(DeploymentConstraint[key])
            except KeyError as e:
                raise ValueError(f"Invalid deployment constraint: {v}") from e
        return out

    def _parse_gpu(self, gpu_value: Any) -> tuple[str | None, int | None]:
        """
        Official spec:
          "gpu": {"name": "NVIDIA L4", "gpu_count": 1}

        Also accepts:
          "gpu": "L40S"
        """
        if gpu_value is None:
            return None, None

        if isinstance(gpu_value, str):
            return gpu_value, None

        if isinstance(gpu_value, dict):
            name = gpu_value.get("id") or gpu_value.get("name")
            gpu_count = gpu_value.get("gpu_count")
            count = int(gpu_count) if gpu_count is not None else None
            return name, count

        raise ValueError(f"Invalid gpu field: {gpu_value!r}")

    def _parse_lmlite_models(self, spec: dict[str, Any]) -> list[LmLiteModelConfig]:
        raw = spec.get("models") or []
        if not raw:
            raise ValueError("lm_lite runtime spec requires non-empty models list")

        models: list[LmLiteModelConfig] = []
        for m in raw:
            if not isinstance(m, dict):
                raise ValueError("Each lm_lite model entry must be an object")

            model_id = m.get("id") or m.get("model_id")
            if not model_id:
                raise ValueError("lm_lite model entry missing 'model_id'/'id'")

            models.append(
                LmLiteModelConfig(
                    id=model_id,
                    max_model_len=int(m.get("max_model_len", 1024)),
                    max_model_concurrency=int(m.get("max_model_concurrency", 1)),
                    model_batch_execute_timeout_ms=int(
                        m.get("model_batch_execute_timeout_ms", 500)
                    ),
                )
            )
        return models

    def _parse_vllm_models(self, spec: dict[str, Any]) -> list[VLLMModelConfig]:
        raw = spec.get("models") or []
        if not raw:
            raise ValueError("vllm runtime spec requires non-empty models list")

        out: list[VLLMModelConfig] = []
        for m in raw:
            if not isinstance(m, dict):
                raise ValueError("Each vllm model entry must be an object")

            d = dict(m)
            if "id" not in d and "model_id" in d:
                d["id"] = d.pop("model_id")
            out.append(VLLMModelConfig(**d))
        return out

    # ---------- lookups ----------
    def _get_transform_spec(self, id: str) -> dict[str, Any]:
        try:
            return self.transforms_by_id[id]
        except KeyError as e:
            raise KeyError(f"Unknown transform id: {id}") from e

    def _get_runtime_for_transform_spec(self, t_spec: dict[str, Any]) -> Any:
        compute_hash = t_spec.get("compute_hash")
        if not compute_hash:
            raise ValueError(f"Transform missing compute_hash: {t_spec}")
        try:
            return self.runtime_instances[compute_hash]
        except KeyError as e:
            raise KeyError(
                f"No runtime instance for compute_hash={compute_hash}"
            ) from e
