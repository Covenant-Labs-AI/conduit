import threading
from abc import ABC, abstractmethod
import asyncio
import json
from typing import List, Type, Any, overload, TypeVar
from conduit.conduit_types import (
    ModelConfig,
    ComputeProvider,
    GPUS,
    NodeStatus,
    Runtime,
    DeploymentType,
    DeploymentStatus,
)
from conduit.utils import (
    best_gpu_for_all_models,
    can_gpu_host_models,
    compute_deployment_key,
    calculate_container_size_gb,
    parse_llm_json,
    dataclass_to_dict,
)
from conduit.state.deployment import (
    get_deployment,
    create_deployment,
    delete_deployment_by_key,
    list_deployments,
    update_deployment_status,
)
from conduit.state.node import (
    get_ready_nodes_by_deployment,
    update_node_info,
    update_node_status,
    create_node,
)
from conduit.state.db import Node
from conduit.compute_provider import (
    EnvConfig,
    start_container_provision,
    wait_node_provision,
    deprovision,
)
from conduit.conduit_http import healthcheck, OpenAIMessage, inf_open_ai_compat
from conduit.mdl import build_mdl_system_prompt


TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class BaseRuntimeBlock(ABC):
    """
    Base class that encapsulates:
      - GPU selection / validation
      - deployment lifecycle (provision, delete, gc)
      - node round-robin + health

    Subclasses are responsible for:
      - runtime-specific image / ports
      - env + container size
      - inference
    """

    # Each subclass gets its own registry, but the attribute is defined here
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

    @abstractmethod
    def build_env(self) -> EnvConfig:
        """
        Return the environment configuration (e.g. MODELS, runtime flags, etc.)
        """
        ...

    @abstractmethod
    def calculate_container_size_gb(self) -> int:
        """
        Return the container size to request (in GB).
        """
        ...

    def __init__(
        self,
        models: List[ModelConfig],
        compute_provider: ComputeProvider,
        gpu: GPUS | None = None,
        replicas: int = 1,
        compute_provider_config_overrides: dict | None = None,
    ) -> None:
        self.models = models
        self.compute_provider = compute_provider
        self.gpu = gpu
        self.replicas = replicas
        self.compute_provider_config_overrides = compute_provider_config_overrides or {}

        self._rr_index = 0
        self._rr_lock = threading.Lock()

        self._init_gpu()

        self.deployment_hash = compute_deployment_key(self)
        deployment = get_deployment(self.deployment_hash)

        self.__class__.BLOCK_REGISTRY.append(self)

        if not deployment:
            self.env: EnvConfig = self.build_env()
            self.container_size_gb = self.calculate_container_size_gb()

            print("🚀 Provisioning... please wait.")
            asyncio.run(self._run_provision_async())
        else:
            self.deployment = deployment

    def _init_gpu(self) -> None:
        if not self.gpu:
            vram_map = best_gpu_for_all_models(self.models)
            best_gpu = vram_map.get("gpu")
            if not best_gpu:
                raise RuntimeError(
                    "No compatible or space on known GPU's found to run your model(s)"
                )

            print(f"Selected GPU: {best_gpu} for models: {self.models}")
            self.gpu = best_gpu
        else:
            vram_map = can_gpu_host_models(self.gpu, self.models)
            if vram_map is None:
                raise ValueError(f"not enough space on {self.gpu} to host models")

    def _next_node_round_robin(self) -> Node:
        nodes = get_ready_nodes_by_deployment(self.deployment.id)
        if not nodes:
            raise RuntimeError(f"No ready nodes for deployment {self.deployment.id}")

        with self._rr_lock:
            node = nodes[self._rr_index % len(nodes)]
            self._rr_index += 1

        return node

    def delete(self) -> None:
        if getattr(self, "deployment", None):
            deployment_nodes = get_ready_nodes_by_deployment(self.deployment.id)
            for node in deployment_nodes:
                deprovision(self.deployment.provider, node.external_id)

            delete_deployment_by_key(self.deployment.deployment_key)

    @classmethod
    def gc(cls) -> None:
        """
        Reconcile all deployments for this runtime subclass.
        """
        desired_keys: set[str] = {
            compute_deployment_key(block) for block in cls.BLOCK_REGISTRY
        }

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
        # For subclasses that override .runtime as a property
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

        for node in self.deployment.nodes:
            port = node.resolve_port(8000)
            result = healthcheck(node.ip_address, port)
            results[str(node.id)] = result

            node_ready = isinstance(result, dict) and result.get("ready") is True
            if node_ready:
                update_node_status(node.id, NodeStatus.DEPLOYED)
            else:
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

    async def _run_provision_async(self):
        self.deployment = create_deployment(
            deployment_key=self.deployment_hash,
            runtime=self.runtime,
            image=self.image,
            gpu=self.gpu,
            deployment_type=self.deployment_type,
            provider=self.compute_provider,
            ports=self.ports,
        )

        async def run_one(replica_idx: int):
            external_id = await asyncio.to_thread(
                start_container_provision,
                self.compute_provider,
                self.image,
                self.ports,
                self.gpu,
                self.env,
                self.container_size_gb,
                self.compute_provider_config_overrides,
            )

            node = create_node(
                external_id=external_id,
                ip_address=None,
                port_map=None,
                deployment_id=self.deployment.id,
            )

            container_info = await asyncio.to_thread(
                wait_node_provision,
                self.compute_provider,
                external_id,
            )

            update_node_info(
                node.id,
                status=NodeStatus.DEPLOYED,
                ip_address=container_info.public_ip,
                port_map=container_info.port_map,
            )

            print(f"  → Node {container_info.id} @ {container_info.public_ip}")
            return container_info

        # Kick off all replicas in parallel
        results = await asyncio.gather(*(run_one(i) for i in range(self.replicas)))

        print(f"✅ Provisioned {len(results)} compute node(s):")

        self.deployment = get_deployment(self.deployment_hash)

        return results


class LMLiteBlock(BaseRuntimeBlock):
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

    def __init__(
        self,
        models: List[ModelConfig],
        compute_provider: ComputeProvider,
        gpu: GPUS | None = None,
        replicas: int = 1,
        compute_provider_config_overrides: dict | None = None,
    ) -> None:
        super().__init__(
            models=models,
            compute_provider=compute_provider,
            gpu=gpu,
            replicas=replicas,
            compute_provider_config_overrides=compute_provider_config_overrides,
        )

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
        input: TIn,
        output: Type[TOut],
    ) -> TOut: ...

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

        if messages and not input and not output:
            res = inf_open_ai_compat(
                node.ip_address,
                port,
                model_id,
                messages,
                guidance,
            )
            return res

        if (input is not None and output is not None) and not messages:
            system_prompt = build_mdl_system_prompt(
                guidance or "",
                input,
                output,
            )
            data_input: List[OpenAIMessage] = [
                {"role": "user", "content": str(dataclass_to_dict(input))}
            ]
            json_response = inf_open_ai_compat(
                node.ip_address,
                port,
                model_id,
                data_input,
                system_prompt,
            )
            return parse_llm_json(json_response, output)

        if (input is not None and output is None) or (
            output is not None and input is None
        ):
            raise ValueError("Both `input` and `output` must be provided together.")

        raise ValueError("Provide either `messages` or (`input`, `output`).")
