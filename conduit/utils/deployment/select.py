import math
from typing import List

from conduit.compute_provider import get_provider_compute_offerings
from conduit.conduit_types import (
    ComputeProvider,
    LmLiteModelConfig,
    Runtime,
    VLLMModelConfig,
)
from conduit.utils import ComputeOffering, gib_to_mib, mib_to_gib
from conduit.utils.accelerators.nvidia import (
    GPUHostingResult,
    GpuInterconnectKind,
    NvidiaGPU,
    architectures_for_dtype,
    build_nvidia_gpus_from_compute_offering,
)
from conduit.utils.model import map_hf_dtype

from .models import ComputeOfferingCandidate, DeploymentConstraint
from .vram import (
    VLLM_OVERHEAD_PER_GPU_GB,
    _next_largest,
    _valid_tp_sizes_from_config,
    call_llm_vram_profile_agent,
    get_single_model_size,
    kv_cache_bytes,
    load_model_index_and_config,
    model_vram_gib_from_total_size,
    try_build_llm_vram_profile_local,
)


def search_compute_providers_for_best_offerings(
    runtime: Runtime,
    provider: ComputeProvider,
    models: List[VLLMModelConfig] | List[LmLiteModelConfig],
    gpu: str | None = None,
    constraints: List[DeploymentConstraint] = [],
) -> List[GPUHostingResult]:

    compute_offerings = get_provider_compute_offerings(provider)
    if gpu:
        offerings = [o for o in compute_offerings if o.id == gpu]
        compute_offerings = offerings

    nvidia_gpus = build_nvidia_gpus_from_compute_offering(compute_offerings)

    if runtime == Runtime.VLLM:
        results = calculate_all_compute_offerings_for_vllm_sorted_by_price(
            compute_offerings, nvidia_gpus, models, constraints
        )
    elif runtime == Runtime.LM_LITE:
        results = calculate_compute_offerings_sorted_by_price(
            compute_offerings, nvidia_gpus, models, constraints
        )
    else:
        raise RuntimeError("Unsupported runtime")

    return results


def calculate_best_compute_offering(
    compute_offerings: List[ComputeOffering],
    nvidia_gpus: List[NvidiaGPU],
    models: List[LmLiteModelConfig],
    constraints: List[DeploymentConstraint],
) -> GPUHostingResult:
    total_param_vram_gb = 0.0
    total_kv_vram_gb = 0.0

    supported_arches = None

    for model in models:
        results = load_model_index_and_config(model.id)
        model_index = results.get("index")

        try:
            model_profile = call_llm_vram_profile_agent(model.id)
        except Exception:
            model_profile = try_build_llm_vram_profile_local(model)

        if model_index:
            total_size: int = model_index["metadata"]["total_size"]
        else:
            total_size = get_single_model_size(model.id)

        max_position_embeddings = model_profile.max_position_embeddings
        if (
            max_position_embeddings is not None
            and model.max_model_len > max_position_embeddings
        ):
            raise ValueError(
                f"Requested context length ({model.max_model_len}) is larger than "
                f"model's max_position_embeddings ({max_position_embeddings}); "
                f"cannot host this model."
            )

        dtype = map_hf_dtype(model_profile.dtype)
        quant_dtype = model_profile.quant_dtype

        if quant_dtype:
            supported_arches = architectures_for_dtype(map_hf_dtype(quant_dtype))
        else:
            supported_arches = architectures_for_dtype(dtype)

        num_kv_heads = model_profile.num_kv_heads
        num_hidden_layers = model_profile.num_hidden_layers
        head_dim = model_profile.head_dim

        # ---- PARAMS (weights) VRAM ----
        model_vram_gb = model_vram_gib_from_total_size(total_size)

        # ---- KV CACHE VRAM ----
        kv_vram_bytes = kv_cache_bytes(
            model.max_model_concurrency,
            model.max_model_len,
            num_hidden_layers,
            num_kv_heads,
            head_dim,
            dtype,
        )
        kv_vram_gb = kv_vram_bytes / (1024**3)

        total_param_vram_gb += model_vram_gb
        total_kv_vram_gb += kv_vram_gb

    raw_total_vram_gb = total_param_vram_gb + total_kv_vram_gb
    required_vram_gb = raw_total_vram_gb
    single_device = DeploymentConstraint.SINGLE_DEVICE in constraints

    candidates: list[ComputeOfferingCandidate] = []
    for offering in compute_offerings:
        gpu = next((u for u in nvidia_gpus if u.name == offering.id), None)

        if (not offering.enterprise_grade) and (
            DeploymentConstraint.ENTERPRISE in constraints
        ):
            continue
        if not gpu:
            continue
        if supported_arches is not None and gpu.architecture not in supported_arches:
            continue

        chip_memory = gpu.memory_free_mib
        n_chips = gib_to_mib(required_vram_gb) / chip_memory
        num_gpu = math.ceil(n_chips)

        if single_device and num_gpu > 1:
            continue
        if num_gpu > offering.max_available:
            continue

        if (
            DeploymentConstraint.HIGH_BANDWIDTH_INTERCONNECT in constraints
            and num_gpu > 1
        ):
            ic = gpu.interconnect
            supports_multi_gpu_domain = (ic.kind != GpuInterconnectKind.NONE) and (
                num_gpu <= ic.max_domain_gpus
            )
            if not supports_multi_gpu_domain:
                continue
        # -------------------------------------------------------

        gpu_prices = math.ceil(n_chips) * offering.price_per_hour
        candidates.append(
            ComputeOfferingCandidate(
                offering=offering, gpu=gpu, num_gpu=num_gpu, price=gpu_prices
            )
        )

    if not candidates:
        raise RuntimeError(
            "No feasible compute offering found for constraints and VRAM requirements."
        )

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


def calculate_compute_offerings_sorted_by_price(
    compute_offerings: List[ComputeOffering],
    nvidia_gpus: List[NvidiaGPU],
    models: List[LmLiteModelConfig],
    constraints: List[DeploymentConstraint],
) -> List[GPUHostingResult]:
    total_param_vram_gb = 0.0
    total_kv_vram_gb = 0.0

    supported_arches = None

    for model in models:
        results = load_model_index_and_config(model.id)
        try:
            model_profile = call_llm_vram_profile_agent(model.id)
        except:
            model_profile = try_build_llm_vram_profile_local(model)

        model_index = results.get("index")

        if model_index:
            total_size: int = model_index["metadata"]["total_size"]
        else:
            total_size = get_single_model_size(model.id)

        max_position_embeddings = model_profile.max_position_embeddings
        if (
            max_position_embeddings is not None
            and model.max_model_len > max_position_embeddings
        ):
            raise ValueError(
                f"Requested context length ({model.max_model_len}) is larger than "
                f"model's max_position_embeddings ({max_position_embeddings}); "
                f"cannot host this model."
            )

        dtype = map_hf_dtype(model_profile.dtype)

        if model_profile.quant_dtype:
            quant_dtype = map_hf_dtype(model_profile.quant_dtype)
            supported_arches = architectures_for_dtype(quant_dtype)
        else:
            supported_arches = architectures_for_dtype(dtype)

        num_kv_heads = model_profile.num_kv_heads
        num_hidden_layers = model_profile.num_hidden_layers
        head_dim = model_profile.head_dim

        if not all(
            isinstance(x, int) for x in [num_kv_heads, num_hidden_layers, head_dim]
        ):
            raise ValueError(
                "Model config missing required KV cache fields: "
                "num_key_value_heads, num_hidden_layers, head_dim"
            )

        # ---- PARAMS (weights) VRAM ----
        model_vram_gb = model_vram_gib_from_total_size(total_size)

        # ---- KV CACHE VRAM ----
        kv_vram_bytes = kv_cache_bytes(
            model.max_model_concurrency,
            model.max_model_len,
            num_hidden_layers,
            num_kv_heads,
            head_dim,
            dtype,
        )
        kv_vram_gb = kv_vram_bytes / (1024**3)

        total_param_vram_gb += model_vram_gb
        total_kv_vram_gb += kv_vram_gb

    raw_total_vram_gb = total_param_vram_gb + total_kv_vram_gb
    required_vram_gb = raw_total_vram_gb
    single_device = DeploymentConstraint.SINGLE_DEVICE in constraints

    candidates: list[ComputeOfferingCandidate] = []
    for offering in compute_offerings:
        gpu = next((u for u in nvidia_gpus if u.name == offering.id), None)

        if (not offering.enterprise_grade) and (
            DeploymentConstraint.ENTERPRISE in constraints
        ):
            continue
        if not gpu:
            continue
        if supported_arches is not None and gpu.architecture not in supported_arches:
            continue

        chip_memory = gpu.memory_free_mib
        n_chips = gib_to_mib(required_vram_gb) / chip_memory
        num_gpu = math.ceil(n_chips)

        if single_device and num_gpu > 1:
            continue
        if num_gpu > offering.max_available:
            continue

        if (
            DeploymentConstraint.HIGH_BANDWIDTH_INTERCONNECT in constraints
            and num_gpu > 1
        ):
            ic = gpu.interconnect
            supports_multi_gpu_domain = (ic.kind != GpuInterconnectKind.NONE) and (
                num_gpu <= ic.max_domain_gpus
            )
            if not supports_multi_gpu_domain:
                continue

        gpu_prices = math.ceil(n_chips) * offering.price_per_hour
        candidates.append(
            ComputeOfferingCandidate(
                offering=offering, gpu=gpu, num_gpu=num_gpu, price=gpu_prices
            )
        )

    candidates_sorted = sorted(candidates, key=lambda c: int(c.price))

    hosting_results: list[GPUHostingResult] = []
    for c in candidates_sorted:
        best_offering: ComputeOffering = c.offering
        best_offering_gpu_count = c.num_gpu
        best_gpu: NvidiaGPU = c.gpu
        headroom_gb = mib_to_gib(best_gpu.memory_mib) - required_vram_gb
        best_gpu.gpu_count = best_offering_gpu_count

        hosting_results.append(
            GPUHostingResult(
                gpus=best_gpu,
                compute_offering=best_offering,
                required_vram_gb=required_vram_gb,
                raw_model_vram_gb=raw_total_vram_gb,
                total_capacity_gb=mib_to_gib(best_gpu.memory_mib),
                headroom_gb=headroom_gb,
                price_cents=best_offering.price_per_hour,  # POTENTAL BUG?
                can_host=True,
            )
        )

    if not hosting_results:
        raise RuntimeError(
            "No feasible compute offering found for constraints and VRAM requirements."
        )

    return hosting_results


def can_nvidia_gpus_host_models(
    gpus: NvidiaGPU,
    models: List[LmLiteModelConfig],
) -> GPUHostingResult:
    """
    Non-raising variant of `can_nvidia_gpus_host_models_or_raise`.
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

            if gpus.architecture not in supported_arches:
                return _fail(
                    f"{gpus.architecture} not supported "
                    f"supported architectures: {supported_arches}"
                )

            num_kv_heads = config.get("num_key_value_heads")
            num_hidden_layers = config.get("num_hidden_layers")
            head_dim = config.get("head_dim")

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

            total_param_vram_gb += model_vram_gb
            total_kv_vram_gb += kv_vram_gb

    except Exception as e:
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
    Raising feasibility check (multi-GPU list).
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

        total_param_vram_gb += model_vram_gb
        total_kv_vram_gb += kv_vram_gb

    raw_total_vram_gb = total_param_vram_gb + total_kv_vram_gb
    required_vram_gb = raw_total_vram_gb
    total_capacity_gb = total_gpu_free_capacity_gb

    if required_vram_gb > total_gpu_free_capacity_gb:
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


def can_local_nvidia_run_vllm_model_or_raise(
    gpus: NvidiaGPU,
    models: List[VLLMModelConfig],
    constraints: List[DeploymentConstraint],
    *,
    overhead_per_gpu_gb: float = VLLM_OVERHEAD_PER_GPU_GB,
) -> GPUHostingResult:
    """
    Raising feasibility check for running a single vLLM model on a local Nvidia GPU pool.
    """
    if not gpus or gpus.gpu_count <= 0:
        raise ValueError(
            "can_local_nvidia_run_vllm_model_or_raise requires at least one GPU"
        )

    free_per_gpu_gb = mib_to_gib(gpus.memory_free_mib)
    model = models[0]

    results = load_model_index_and_config(model.id)
    model_profile = call_llm_vram_profile_agent(model.id)

    model_index = results.get("index")
    if model_index:
        total_size: int = model_index["metadata"]["total_size"]
    else:
        total_size = get_single_model_size(model.id)

    max_position_embeddings = model_profile.max_position_embeddings
    if (
        max_position_embeddings is not None
        and model.max_model_len > max_position_embeddings
    ):
        raise ValueError(
            f"Requested context length ({model.max_model_len}) is larger than "
            f"model's max_position_embeddings ({max_position_embeddings}); cannot host this model."
        )

    dtype = map_hf_dtype(model_profile.dtype)
    if model_profile.quant_dtype:
        quant_dtype = map_hf_dtype(model_profile.quant_dtype)
        supported_arches = architectures_for_dtype(quant_dtype)
    else:
        supported_arches = architectures_for_dtype(dtype)

    if gpus.architecture not in supported_arches:
        raise ValueError(
            f"{gpus.architecture} not supported; supported architectures: {supported_arches}"
        )

    num_kv_heads = model_profile.num_kv_heads
    num_hidden_layers = model_profile.num_hidden_layers
    head_dim = model_profile.head_dim

    if not all(isinstance(x, int) for x in [num_kv_heads, num_hidden_layers, head_dim]):
        raise ValueError(
            "Model profile missing required KV cache fields: "
            "num_key_value_heads, num_hidden_layers, head_dim"
        )

    model_vram_gb = model_vram_gib_from_total_size(total_size)

    kv_vram_bytes = kv_cache_bytes(
        model.max_num_seqs,
        model.max_model_len,
        num_hidden_layers,
        num_kv_heads,
        head_dim,
        dtype,
    )
    kv_vram_gb = kv_vram_bytes / (1024**3)

    raw_total_vram_gb = model_vram_gb + kv_vram_gb

    valid_tps = _valid_tp_sizes_from_config(model_profile)
    if not valid_tps:
        raise RuntimeError(
            "Could not determine valid tensor-parallel sizes for this model"
        )

    single_device = DeploymentConstraint.SINGLE_DEVICE in constraints
    require_hbi = DeploymentConstraint.HIGH_BANDWIDTH_INTERCONNECT in constraints

    max_tp_allowed = min(gpus.gpu_count, getattr(gpus, "max_available", gpus.gpu_count))
    if single_device:
        max_tp_allowed = min(max_tp_allowed, 1)

    if overhead_per_gpu_gb >= free_per_gpu_gb:
        raise RuntimeError(
            f"Per-GPU overhead ({overhead_per_gpu_gb:.2f} GiB) >= free VRAM per GPU "
            f"({free_per_gpu_gb:.2f} GiB); cannot run vLLM."
        )

    est = max(1, math.ceil(raw_total_vram_gb / (free_per_gpu_gb - overhead_per_gpu_gb)))
    est = min(est, max_tp_allowed)

    def step_to_next_valid(tp_value: int) -> int | None:
        if tp_value in valid_tps:
            return tp_value
        return _next_largest(valid_tps, tp_value)

    tp = step_to_next_valid(est)
    if tp is None:
        raise RuntimeError(
            f"No valid TP size >= {est} for this model (valid: {sorted(valid_tps)})"
        )

    chosen_tp: int | None = None

    while True:
        if tp is None or tp > max_tp_allowed:
            break
        if tp not in valid_tps:
            tp = _next_largest(valid_tps, tp)
            continue

        if require_hbi and tp > 1:
            ic = gpus.interconnect
            supports_tp = (ic.kind != GpuInterconnectKind.NONE) and (
                tp <= ic.max_domain_gpus
            )
            if not supports_tp:
                tp = _next_largest(valid_tps, tp)
                continue

        required_per_gpu_gb = (raw_total_vram_gb / tp) + overhead_per_gpu_gb
        if required_per_gpu_gb <= free_per_gpu_gb:
            chosen_tp = tp
            break

        tp = _next_largest(valid_tps, tp)

    if chosen_tp is None:
        raise RuntimeError(
            "No feasible tensor-parallel size found for vLLM given VRAM + constraints. "
            f"(raw_total_vram_gb={raw_total_vram_gb:.2f} GiB, "
            f"free_per_gpu_gb={free_per_gpu_gb:.2f} GiB, "
            f"overhead_per_gpu_gb={overhead_per_gpu_gb:.2f} GiB, "
            f"valid_tps={sorted(valid_tps)}, "
            f"gpu_count={gpus.gpu_count}, "
            f"require_hbi={require_hbi})."
        )

    required_vram_gb = raw_total_vram_gb + (overhead_per_gpu_gb * chosen_tp)
    total_capacity_gb = mib_to_gib(gpus.memory_mib) * chosen_tp
    headroom_gb = total_capacity_gb - required_vram_gb

    gpus_for_result = gpus
    gpus_for_result.gpu_count = chosen_tp

    return GPUHostingResult(
        gpus=gpus_for_result,
        price_cents=0,
        required_vram_gb=required_vram_gb,
        raw_model_vram_gb=raw_total_vram_gb,
        total_capacity_gb=total_capacity_gb,
        headroom_gb=headroom_gb,
        can_host=True,
    )


def calculate_best_compute_offering_for_vllm(
    compute_offerings: List[ComputeOffering],
    nvidia_gpus: List[NvidiaGPU],
    models: List[VLLMModelConfig],
    constraints: List[DeploymentConstraint],
    *,
    overhead_per_gpu_gb: float = VLLM_OVERHEAD_PER_GPU_GB,
) -> GPUHostingResult:
    """
    vLLM-focused offering selection.
    """
    if len(models) != 1:
        raise ValueError(
            "vLLM supports exactly one model per deployment; pass a single VLLMModelConfig."
        )

    model = models[0]

    results = load_model_index_and_config(model.id)
    try:
        model_profile = call_llm_vram_profile_agent(model.id)
    except:
        model_profile = try_build_llm_vram_profile_local(model)

    model_index = results.get("index")
    if model_index:
        total_size: int = model_index["metadata"]["total_size"]
    else:
        total_size = get_single_model_size(model.id)

    max_position_embeddings = model_profile.max_position_embeddings
    if (
        max_position_embeddings is not None
        and model.max_model_len > max_position_embeddings
    ):
        raise ValueError(
            f"Requested context length ({model.max_model_len}) is larger than "
            f"model's max_position_embeddings ({max_position_embeddings}); cannot host this model."
        )

    dtype = map_hf_dtype(model_profile.dtype)

    if model_profile.quant_dtype:
        quant_dtype = map_hf_dtype(model_profile.quant_dtype)
        supported_arches = architectures_for_dtype(quant_dtype)
    else:
        supported_arches = architectures_for_dtype(dtype)

    num_kv_heads = model_profile.num_kv_heads
    num_hidden_layers = model_profile.num_hidden_layers
    head_dim = model_profile.head_dim

    if not all(isinstance(x, int) for x in [num_kv_heads, num_hidden_layers, head_dim]):
        raise ValueError(
            "Model config missing required KV cache fields: "
            "num_key_value_heads, num_hidden_layers, head_dim"
        )

    model_vram_gb = model_vram_gib_from_total_size(total_size)

    kv_vram_bytes = kv_cache_bytes(
        model.max_num_seqs,
        model.max_model_len,
        num_hidden_layers,
        num_kv_heads,
        head_dim,
        dtype,
    )
    kv_vram_gb = kv_vram_bytes / (1024**3)

    raw_total_vram_gb = model_vram_gb + kv_vram_gb
    valid_tps = _valid_tp_sizes_from_config(model_profile)

    single_device = DeploymentConstraint.SINGLE_DEVICE in constraints
    candidates: list[ComputeOfferingCandidate] = []

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

        free_per_gpu_gb = mib_to_gib(gpu.memory_free_mib)
        chip_budget_gb = free_per_gpu_gb

        est = max(1, math.ceil(raw_total_vram_gb / chip_budget_gb))

        def step_to_next_valid(tp_value: int) -> int | None:
            if tp_value in valid_tps:
                return tp_value
            return _next_largest(valid_tps, tp_value)

        tp = step_to_next_valid(est) or est

        feasible = False
        while True:
            if single_device and tp > 1:
                feasible = False
                break
            if tp > offering.max_available:
                feasible = False
                break
            if tp not in valid_tps:
                nxt = _next_largest(valid_tps, tp)
                if nxt is None:
                    feasible = False
                    break
                tp = nxt
                continue

            if (
                DeploymentConstraint.HIGH_BANDWIDTH_INTERCONNECT in constraints
                and tp > 1
            ):
                ic = gpu.interconnect
                supports_tp = (ic.kind != GpuInterconnectKind.NONE) and (
                    tp <= ic.max_domain_gpus
                )
                if not supports_tp:
                    nxt = _next_largest(valid_tps, tp)
                    if nxt is None:
                        feasible = False
                        break
                    tp = nxt
                    continue

            required_per_gpu_gb = (raw_total_vram_gb / tp) + overhead_per_gpu_gb
            if required_per_gpu_gb <= chip_budget_gb:
                feasible = True
                break

            nxt = _next_largest(valid_tps, tp)
            if nxt is None:
                feasible = False
                break
            tp = nxt

        if not feasible:
            continue

        gpu_prices = tp * offering.price_per_hour
        candidates.append(
            ComputeOfferingCandidate(
                offering=offering, gpu=gpu, num_gpu=tp, price=gpu_prices
            )
        )

    if not candidates:
        raise RuntimeError(
            "No feasible compute offering found for vLLM constraints and VRAM requirements."
        )

    candidates_sorted = sorted(candidates, key=lambda c: int(c.price))
    best = candidates_sorted[0]
    best_offering = best.offering
    best_gpu = best.gpu
    best_tp = best.num_gpu

    required_vram_gb = raw_total_vram_gb + (overhead_per_gpu_gb * best_tp)
    total_capacity_gb = mib_to_gib(best_gpu.memory_mib) * best_tp
    headroom_gb = total_capacity_gb - required_vram_gb

    best_gpu.gpu_count = best_tp

    return GPUHostingResult(
        gpus=best_gpu,
        compute_offering=best_offering,
        required_vram_gb=required_vram_gb,
        raw_model_vram_gb=raw_total_vram_gb,
        total_capacity_gb=total_capacity_gb,
        headroom_gb=headroom_gb,
        price_cents=best_offering.price_per_hour * best_gpu.gpu_count,
        can_host=True,
    )


def calculate_all_compute_offerings_for_vllm_sorted_by_price(
    compute_offerings: List[ComputeOffering],
    nvidia_gpus: List[NvidiaGPU],
    models: List[VLLMModelConfig],
    constraints: List[DeploymentConstraint],
    *,
    overhead_per_gpu_gb: float = VLLM_OVERHEAD_PER_GPU_GB,
) -> List[GPUHostingResult]:
    """
    vLLM-focused offering selection.
    Returns a list of GPUHostingResult sorted by price (cheapest first).
    """
    if len(models) != 1:
        raise ValueError(
            "vLLM supports exactly one model per deployment; pass a single VLLMModelConfig."
        )

    model = models[0]

    results = load_model_index_and_config(model.id)
    try:
        model_profile = call_llm_vram_profile_agent(model.id)
    except:
        model_profile = try_build_llm_vram_profile_local(model)

    model_index = results.get("index")
    if model_index:
        total_size: int = model_index["metadata"]["total_size"]
    else:
        total_size = get_single_model_size(model.id)

    max_position_embeddings = model_profile.max_position_embeddings
    if (
        max_position_embeddings is not None
        and model.max_model_len > max_position_embeddings
    ):
        raise ValueError(
            f"Requested context length ({model.max_model_len}) is larger than "
            f"model's max_position_embeddings ({max_position_embeddings}); cannot host this model."
        )

    dtype = map_hf_dtype(model_profile.dtype)

    if model_profile.quant_dtype:
        quant_dtype = map_hf_dtype(model_profile.quant_dtype)
        supported_arches = architectures_for_dtype(quant_dtype)
    else:
        supported_arches = architectures_for_dtype(dtype)

    num_kv_heads = model_profile.num_kv_heads
    num_hidden_layers = model_profile.num_hidden_layers
    head_dim = model_profile.head_dim

    if not all(isinstance(x, int) for x in [num_kv_heads, num_hidden_layers, head_dim]):
        raise ValueError(
            "Model config missing required KV cache fields: "
            "num_key_value_heads, num_hidden_layers, head_dim"
        )

    model_vram_gb = model_vram_gib_from_total_size(total_size)

    kv_vram_bytes = kv_cache_bytes(
        model.max_num_seqs,
        model.max_model_len,
        num_hidden_layers,
        num_kv_heads,
        head_dim,
        dtype,
    )
    kv_vram_gb = kv_vram_bytes / (1024**3)

    raw_total_vram_gb = model_vram_gb + kv_vram_gb
    valid_tps = _valid_tp_sizes_from_config(model_profile)

    single_device = DeploymentConstraint.SINGLE_DEVICE in constraints
    hosting_results: list[GPUHostingResult] = []

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

        free_per_gpu_gb = mib_to_gib(gpu.memory_free_mib)
        chip_budget_gb = free_per_gpu_gb

        est = max(1, math.ceil(raw_total_vram_gb / chip_budget_gb))

        def step_to_next_valid(tp_value: int) -> int | None:
            if tp_value in valid_tps:
                return tp_value
            return _next_largest(valid_tps, tp_value)

        tp = step_to_next_valid(est) or est

        feasible = False
        while True:
            if single_device and tp > 1:
                feasible = False
                break
            if tp > offering.max_available:
                feasible = False
                break
            if tp not in valid_tps:
                nxt = _next_largest(valid_tps, tp)
                if nxt is None:
                    feasible = False
                    break
                tp = nxt
                continue

            if (
                DeploymentConstraint.HIGH_BANDWIDTH_INTERCONNECT in constraints
                and tp > 1
            ):
                ic = gpu.interconnect
                supports_tp = (ic.kind != GpuInterconnectKind.NONE) and (
                    tp <= ic.max_domain_gpus
                )
                if not supports_tp:
                    nxt = _next_largest(valid_tps, tp)
                    if nxt is None:
                        feasible = False
                        break
                    tp = nxt
                    continue

            required_per_gpu_gb = (raw_total_vram_gb / tp) + overhead_per_gpu_gb
            if required_per_gpu_gb <= chip_budget_gb:
                feasible = True
                break

            nxt = _next_largest(valid_tps, tp)
            if nxt is None:
                feasible = False
                break
            tp = nxt

        if not feasible:
            continue

        required_vram_gb = raw_total_vram_gb + (overhead_per_gpu_gb * tp)
        total_capacity_gb = mib_to_gib(gpu.memory_mib) * tp
        headroom_gb = total_capacity_gb - required_vram_gb

        gpu.gpu_count = tp

        hosting_results.append(
            GPUHostingResult(
                gpus=gpu,
                compute_offering=offering,
                required_vram_gb=required_vram_gb,
                raw_model_vram_gb=raw_total_vram_gb,
                total_capacity_gb=total_capacity_gb,
                headroom_gb=headroom_gb,
                price_cents=offering.price_per_hour * gpu.gpu_count,
                can_host=True,
            )
        )

    if not hosting_results:
        raise RuntimeError(
            "No feasible compute offering found for vLLM constraints and VRAM requirements."
        )

    hosting_results_sorted = sorted(hosting_results, key=lambda r: int(r.price_cents))
    return hosting_results_sorted
