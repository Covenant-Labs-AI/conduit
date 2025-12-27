import os
import threading
from abc import ABC, abstractmethod
import asyncio
import json
from typing import List, Type, Any, overload, TypeVar
from conduit.conduit_types import (
    LmLiteModelConfig,
    ComputeProvider,
    NodeStatus,
    Runtime,
    DeploymentType,
    DeploymentStatus,
)
from conduit.utils import dataclass_to_dict, gib_to_bytes
from conduit.utils.accelerators.nvidia import (
    GPUHostingResult,
    NvidiaGPU,
    build_nvidia_gpus_from_compute_offering,
    detect_nvidia,
)
from conduit.utils.deployment import (
    DeploymentConstraint,
    compute_deployment_key,
    calculate_best_compute_offering,
    can_nvidia_gpus_host_models,
    calculate_container_size_gb,
    gpu_id_and_count_label,
    has_enough_disk_space,
    parse_llm_json,
)
from conduit.state.deployment import (
    get_deployment,
    create_deployment,
    delete_deployment_by_key,
    list_deployments,
    update_deployment_status,
)
from conduit.state.node import (
    get_nodes_by_deployment,
    get_ready_nodes_by_deployment,
    update_node_status,
    create_node,
)
from conduit.state.db import Node, get_session
from conduit.compute_provider import (
    EnvConfig,
    get_provider_compute_offerings,
    restart_provider_node,
    start_container_provision,
    stop_provider_node,
    wait_node_provision,
    deprovision,
)
from conduit.conduit_http import healthcheck, OpenAIMessage, inf_open_ai_compat
from conduit.mdl import build_mdl_system_prompt


TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class BaseRuntimeBlock(ABC):
    """
    Provisioning / lifecycle only:
      - deployment create/restart/stop/delete/gc
      - node health -> deployment readiness

    """

    BLOCK_REGISTRY: list["BaseRuntimeBlock"] = []

    @property
    @abstractmethod
    def runtime(self) -> Runtime: ...

    @property
    @abstractmethod
    def deployment_type(self) -> DeploymentType: ...

    @property
    @abstractmethod
    def image(self) -> str: ...

    @property
    @abstractmethod
    def ports(self) -> str: ...

    def __init__(
        self,
        *,
        compute_provider: ComputeProvider,
        gpu_calc: GPUHostingResult,
        replicas: int,
        env: EnvConfig,
        container_size_gb: int,
        compute_provider_config_overrides: dict | None = None,
    ) -> None:
        self.compute_provider = compute_provider
        self.gpu_calc = gpu_calc
        self.replicas = replicas
        self.env = env
        self.container_size_gb = container_size_gb
        self.compute_provider_config_overrides = compute_provider_config_overrides or {}

        if self.gpu_calc:
            self.gpu_id, self.gpu_str_repr, self.num_gpus = gpu_id_and_count_label(
                self.gpu_calc.gpus
            )

        self.__class__.BLOCK_REGISTRY.append(self)

        if not self.deployment:
            if not gpu_calc:
                raise RuntimeError("No GPU calculation aborting")
            if not gpu_calc.can_host:
                raise RuntimeError(
                    "Insufficient GPU VRAM for this model. "
                    f"Required: {gpu_calc.required_vram_gb:.1f} GB (raw model: {gpu_calc.raw_model_vram_gb:.1f} GB). "
                    f"Available: {gpu_calc.total_capacity_gb:.1f} GB across {gpu_calc.gpus} GPU(s) "
                    f"(headroom: {gpu_calc.headroom_gb:.1f} GB). "
                    "Choose a larger GPU / add GPUs, reduce context/batch size, or use quantization."
                )

            if compute_provider == ComputeProvider.LOCAL:
                if not has_enough_disk_space(gib_to_bytes(self.container_size_gb)):
                    raise OSError(
                        f"Not enough disk space in {os.getcwd()} "
                        f"(need {self.container_size_gb:.2f} GiB free)."
                    )

            print("🚀 Provisioning... please wait.")
            asyncio.run(self._run_provision_async())
        else:
            if self.deployment.status == DeploymentStatus.STOPPED:
                print("♻️ Restarting stopped deployment...")
                asyncio.run(self._run_restart_async())

    def stop(self) -> None:
        deployment_nodes = get_nodes_by_deployment(self.deployment.id)
        for node in deployment_nodes:
            stop_provider_node(self.deployment.provider, node.external_id)
            update_node_status(node.id, NodeStatus.STOPPED)

        update_deployment_status(self.deployment.id, DeploymentStatus.STOPPED)

    def restart(self) -> None:
        deployment_nodes = get_nodes_by_deployment(self.deployment.id)
        for node in deployment_nodes:
            restart_provider_node(self.deployment.provider, node.external_id)
            update_node_status(node.id, NodeStatus.DEPLOYED)

    def delete(self) -> None:
        if getattr(self, "deployment", None):
            deployment_nodes = get_nodes_by_deployment(self.deployment.id)
            for node in deployment_nodes:
                deprovision(self.deployment.provider, node.external_id)
            delete_deployment_by_key(self.deployment.deployment_key)

    @classmethod
    def gc(cls) -> None:
        desired_keys: set[str] = {b.deployment_hash for b in cls.BLOCK_REGISTRY}

        existing_deployments = list_deployments(
            runtime=cls._runtime_for_gc(),
            deployment_type=cls._deployment_type_for_gc(),
        )

        for deployment in existing_deployments:
            if deployment.deployment_key in desired_keys:
                continue

            deployment_nodes = get_ready_nodes_by_deployment(deployment.id)
            for node in deployment_nodes:
                deprovision(deployment.provider, node.external_id)

            delete_deployment_by_key(deployment.deployment_key)

    @classmethod
    def _runtime_for_gc(cls) -> Runtime:
        sample = next(
            (b for b in cls.BLOCK_REGISTRY if getattr(b, "runtime", None)), None
        )
        if sample is not None:
            return sample.runtime
        raise RuntimeError("Runtime not available for GC; override _runtime_for_gc.")

    @classmethod
    def _deployment_type_for_gc(cls) -> DeploymentType:
        sample = next(
            (b for b in cls.BLOCK_REGISTRY if getattr(b, "deployment_type", None)), None
        )
        if sample is not None:
            return sample.deployment_type
        raise RuntimeError(
            "Deployment type not available for GC; override _deployment_type_for_gc."
        )

    def health(self) -> dict:
        results = {}
        all_ready = True
        deployment_nodes = get_nodes_by_deployment(self.deployment.id)

        for node in deployment_nodes:
            port = node.resolve_port(8000)
            result = healthcheck(node.ip_address, port)
            results[str(node.id)] = result

            node_ready = isinstance(result, dict) and result.get("ready") is True
            update_node_status(
                node.id, NodeStatus.DEPLOYED if node_ready else NodeStatus.PROVISIONING
            )
            if not node_ready:
                all_ready = False

        if all_ready:
            update_deployment_status(self.deployment.id, DeploymentStatus.DEPLOYED)

        return results

    @property
    def ready(self) -> bool:
        return all(
            isinstance(data, dict) and data.get("ready") is True
            for data in self.health().values()
        )

    async def _run_restart_async(self):
        deployment_nodes = get_nodes_by_deployment(self.deployment.id)

        async def restart_one(node):
            await asyncio.to_thread(
                restart_provider_node,
                self.compute_provider,
                node.external_id,
            )
            container_info = await asyncio.to_thread(
                wait_node_provision,
                self.compute_provider,
                node.external_id,
            )

            update_node_status(node.id, status=NodeStatus.DEPLOYED)
            update_deployment_status(self.deployment.id, DeploymentStatus.DEPLOYED)

            print(
                f"  ↻ Restarted node {container_info.id} @ {container_info.public_ip}"
            )
            return container_info

        return await asyncio.gather(*(restart_one(n) for n in deployment_nodes))

    async def _run_provision_async(self):
        async def provision_one(i: int):
            await asyncio.sleep(i * 0.5)
            external_id = None
            try:
                external_id = await asyncio.to_thread(
                    start_container_provision,
                    self.compute_provider,
                    self.gpu_calc,
                    self.image,
                    self.ports,
                    self.gpu_id,
                    self.num_gpus,
                    self.env,
                    self.container_size_gb,
                    self.compute_provider_config_overrides,
                )
                info = await asyncio.to_thread(
                    wait_node_provision, self.compute_provider, external_id
                )
                return i, external_id, info, None
            except BaseException as e:
                # external_id will be set if start_container_provision succeeded
                return i, external_id, None, e

        results = await asyncio.gather(
            *(provision_one(i) for i in range(self.replicas))
        )
        failures = [r for r in results if r[3] is not None]
        if failures:
            started_external_ids = [
                external_id for _, external_id, _, _ in results if external_id
            ]
            if started_external_ids:
                await asyncio.gather(
                    *(
                        asyncio.to_thread(deprovision, self.compute_provider, ext_id)
                        for ext_id in started_external_ids
                    ),
                    return_exceptions=True,
                )
            i, external_id, _, err = failures[0]
            msg = f"Provisioning failed (replica={i}, external_id={external_id})"
            raise RuntimeError(msg) from err

        with get_session() as s:
            with s.begin():
                self.deployment = create_deployment(
                    session=s,
                    deployment_key=self.deployment_hash,
                    runtime=self.runtime,
                    image=self.image,
                    gpu=self.gpu_str_repr,
                    deployment_type=self.deployment_type,
                    provider=self.compute_provider,
                    ports=self.ports,
                    replicas=self.replicas,
                )

                for _, external_id, container_info, _ in results:
                    create_node(
                        session=s,
                        external_id=external_id,  # NOT NULL satisfied
                        ip_address=container_info.public_ip,
                        port_map=container_info.port_map,
                        deployment_id=self.deployment.id,
                    )

                self.deployment.status = DeploymentStatus.DEPLOYED

            s.refresh(self.deployment)
            return results


class LmInferenceBlock(BaseRuntimeBlock):
    """
    Adds LLM-specific concerns:
      - models
      - GPU selection/validation for those models
      - node round-robin selection
    """

    def __init__(
        self,
        *,
        models: List["LmLiteModelConfig"],
        compute_provider: ComputeProvider,
        gpu: str | None = None,
        constraints: List[DeploymentConstraint] = [],
        replicas: int = 1,
        num_gpus: int = 1,
        compute_provider_config_overrides: dict | None = None,
    ) -> None:
        self.models = models

        self._rr_index = 0
        self._rr_lock = threading.Lock()

        self.compute_provider = compute_provider
        self.constraints = constraints
        self.gpu = gpu
        self.num_gpus = num_gpus

        env = self.build_env()
        container_size_gb = self.calculate_container_size_gb()

        self.compute_provider = compute_provider
        self.gpu = gpu
        self.replicas = replicas
        self.compute_provider_config_overrides = compute_provider_config_overrides or {}

        deployment_hash = compute_deployment_key(self)
        self.deployment = get_deployment(deployment_hash)

        if self.deployment:
            host_result = None
        else:
            host_result = self.calculate_model_vram()

        super().__init__(
            compute_provider=compute_provider,
            gpu_calc=host_result,
            replicas=replicas,
            env=env,
            container_size_gb=container_size_gb,
            compute_provider_config_overrides=compute_provider_config_overrides,
        )

    @abstractmethod
    def build_env(self) -> EnvConfig: ...

    @abstractmethod
    def calculate_container_size_gb(self) -> int: ...

    def calculate_model_vram(self) -> GPUHostingResult:
        self.deployment_hash = compute_deployment_key(self)
        if self.compute_provider == ComputeProvider.LOCAL:
            nvidia_gpus = detect_nvidia()
            if not nvidia_gpus:
                raise RuntimeError("No NVIDIA GPU's detected")
            return can_nvidia_gpus_host_models(nvidia_gpus, self.models)

        print(
            f"🔎 Scanning {self.compute_provider} for "
            f"{'available ' + self.gpu if self.gpu else 'available offerings'}..."
        )

        compute_offerings = get_provider_compute_offerings(self.compute_provider)

        if self.gpu:
            offerings = [o for o in compute_offerings if o.id == self.gpu]
            result = (
                f"✅ Found {len(offerings)} compute offerings for {self.gpu}"
                if offerings
                else f"❌ No compute offering found for id={self.gpu}"
            )
        else:
            result = f"✅ Found {len(compute_offerings)} compute offerings"

        print(result)

        nvidia_gpus = build_nvidia_gpus_from_compute_offering(compute_offerings)

        if not self.gpu:
            result = calculate_best_compute_offering(
                compute_offerings, nvidia_gpus, self.models, self.constraints
            )

            print("🧮 Finished capacity calculations.")
            print(
                '"""""""""""""""""""""""""""""""""""""""""""""""""""\n'
                f"  GPU: {result.gpus.name} x{result.gpus.gpu_count} ({result.gpus.memory_mib / 1024:.0f} GB)\n"
                f"  Required VRAM (GB): {result.required_vram_gb:.2f}\n"
                f"  Raw model VRAM (GB): {result.raw_model_vram_gb:.2f}\n"
                f"  Total capacity (GB): {result.total_capacity_gb:.2f}\n"
                f"  Headroom (GB): {result.headroom_gb:.2f}\n"
                f"  Price: ${result.price_cents / 100}/hr\n"
                '"""""""""""""""""""""""""""""""""""""""""""""""""""'
            )

            return result

        selected = next((g for g in nvidia_gpus if g.name == self.gpu), None)
        if selected:
            selected.gpu_count = self.num_gpus
        else:
            raise ValueError("GPU not listed in Provider GPU's")

        offerings = [o for o in compute_offerings if o.id == self.gpu]
        print(offerings)
        nvidia_gpus = build_nvidia_gpus_from_compute_offering(offerings)
        result = calculate_best_compute_offering(
            compute_offerings, nvidia_gpus, self.models, self.constraints
        )

        return result

    def _next_node_round_robin(self) -> Node:
        nodes = get_ready_nodes_by_deployment(self.deployment.id)
        if not nodes:
            raise RuntimeError(f"No ready nodes for deployment {self.deployment.id}")

        with self._rr_lock:
            node = nodes[self._rr_index % len(nodes)]
            self._rr_index += 1

        return node


class LMLiteBlock(LmInferenceBlock):
    _runtime = Runtime.LM_LITE
    _deployment_type = DeploymentType.LLM
    _image = "covenantlab/lmlite"
    _ports = "8000"

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def deployment_type(self) -> DeploymentType:
        return self._deployment_type

    @property
    def image(self) -> str:
        return self._image

    @property
    def ports(self) -> str:
        return self._ports

    def build_env(self) -> EnvConfig:
        return {
            "env": {
                "MODELS": json.dumps([model.__dict__ for model in self.models]),
            }
        }

    def calculate_container_size_gb(self) -> int:
        return calculate_container_size_gb(self.models, runtime_size_gb=3) * 2

    @overload
    def __call__(
        self,
        model_id: str,
        messages: List[OpenAIMessage],
        guidance: str | None = None,
        *,
        output: None = ...,
        input: None = ...,
    ) -> str: ...

    @overload
    def __call__(
        self,
        model_id: str,
        messages: None = ...,
        guidance: str | None = None,
        *,
        input: Any,
        output: Type[Any],
    ) -> Any: ...

    def __call__(
        self,
        model_id: str,
        messages: List[OpenAIMessage] | None = None,
        guidance: str | None = None,
        *,
        input: Any = None,
        output: Type[Any] | None = None,
    ) -> Any:
        node = self._next_node_round_robin()
        port = node.resolve_port(int(self.ports))

        if messages and input is None and output is None:
            return inf_open_ai_compat(
                node.ip_address, port, model_id, messages, guidance
            )

        if (input is not None and output is not None) and not messages:
            system_prompt = build_mdl_system_prompt(guidance or "", input, output)
            data_input: List[OpenAIMessage] = [
                {"role": "user", "content": str(dataclass_to_dict(input))}
            ]
            json_response = inf_open_ai_compat(
                node.ip_address, port, model_id, data_input, system_prompt
            )
            return parse_llm_json(json_response, output)

        if (input is not None) ^ (output is not None):
            raise ValueError("Both `input` and `output` must be provided together.")

        raise ValueError("Provide either `messages` or (`input`, `output`).")
