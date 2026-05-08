from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class DeploymentStatus(Enum):
    DEPLOYING = "DEPLOYING"
    DEPLOYED = "DEPLOYED"
    STOPPED = "STOPPED"


class NodeStatus(Enum):
    PROVISIONING = "PROVISIONING"
    PROVISIONED = "PROVISIONED"
    DEPLOYED = "DEPLOYED"
    STOPPED = "STOPPED"


class Runtime(Enum):
    VLLM = ("VLLM",)
    LM_LITE = "LM_LITE"
    CUSTOM = "CUSTOM"


class ComputeProvider(Enum):
    LOCAL = "LOCAL"
    RUNPOD = "RUNPOD"


class DeploymentType(Enum):
    LLM = "LLM"


@dataclass
class LmModelConfig:
    id: str
    max_model_len: int = 1024
    max_model_concurrency: int = 1
    task: Literal["generate", "embed", "rerank"] = "generate"


@dataclass
class LmLiteModelConfig(LmModelConfig):
    model_batch_execute_timeout_ms: int = 500


@dataclass
class VLLMModelConfig:
    id: str
    max_model_len: int = 1024

    # --- Common engine/model-loading args ---
    tokenizer: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False

    # dtype/quantization
    dtype: str | None = "auto"  # e.g. "auto", "half", "bfloat16"
    quantization: str | None = None  # e.g. "awq", "gptq", etc.
    generation_config: str | None = None  # e.g. "vllm"

    # --- Parallelism / distributed ---
    tensor_parallel_size: int = 1  # `--tensor-parallel-size`
    pipeline_parallel_size: int = 1  # `--pipeline-parallel-size`

    # --- Memory / KV cache / offload ---
    gpu_memory_utilization: float = 0.90  # `--gpu-memory-utilization`
    swap_space_gb: float | None = None  # `--swap-space` (GB)
    cpu_offload_gb: float | None = None  # `--cpu-offload-gb`
    block_size: int | None = None  # engine/cache block size
    num_gpu_blocks_override: int | None = None  # `--num-gpu-blocks-override`

    # --- Batching / scheduling / throughput controls ---
    max_num_batched_tokens: int | None = None  # `--max-num-batched-tokens`
    max_num_seqs: int | None = 1  # `--max-num-seqs`
    enable_chunked_prefill: bool | None = None  # newer versions

    # --- LoRA / adapters ---
    enable_lora: bool = False
    lora_modules: list[str] = field(default_factory=list)
    max_loras: int | None = None
    max_lora_rank: int | None = None

    # --- OpenAI-compatible server (frontend) args ---
    served_model_name: str | None = None  # `--served-model-name`
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None  # `--api-key`

    # CORS
    allowed_origins: list[str] = field(default_factory=list)
    allowed_methods: list[str] = field(default_factory=list)
    allowed_headers: list[str] = field(default_factory=list)

    # Tool calling / function calling support
    enable_auto_tool_choice: bool | None = None
    tool_call_parser: str | None = None
    tool_parser_plugin: str | None = None

    # --- Escape hatches for everything else ---
    engine_extra: dict[str, Any] = field(default_factory=dict)
    server_extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.model:
            raise ValueError("VLLMModelConfig.model must be set (HF repo/path).")
        if self.tensor_parallel_size < 1 or self.pipeline_parallel_size < 1:
            raise ValueError("Parallel sizes must be >= 1.")
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be in (0, 1].")


@dataclass
class LLMRawTrace:
    raw_input: str
    raw_output: str
