# Conduit

**Conduit** is an AI app-building framework for **open-source language models**.  
It gives developers **type-safe**, **high-flexibility** building blocks for deploying and composing AI systems — without locking you into a single model or compute provider.

---

## ✨ Core Concepts

- **Type-Safe AI Apps**
  - Use Python `@dataclass` types for request/response schemas.
  - Conduit automatically generates model-safe type hints and validates payloads.

- **Flexible Deployment**
  - Choose any supported **model**, **GPU**, and **compute provider**.
  - Conduit automatically:
    - Calculates GPU + memory requirements
    - Validates resource compatibility
    - Manages deployment + scaling

- **AI-Native Workflow Blocks**
  - Build pipelines by chaining high-level runtime blocks:
    - `LMLiteBlock` — inference runtime (streaming + batching)
    - `HttpGetBlock`
    - `FileSystemWriteBlock`
    - `Sqlite3Block`
    - …more coming!

---



## 📦 Installation

### Requirements
- **Python 3.12.6** or newer

### Install from GitHub (latest)

```bash
pip install "git+https://github.com/Covenant-Labs-AI/conduit.git@main"

```
## 🔥 LMLite — AI Inference Runtime

Conduit integrates seamlessly with **LMLite** — our multi-model/multi-GPU batching engine (still under development!).  
It handles:

- Efficient batch processing
- Round-robin replicas
- Health-check readiness gates
- GPU-aware execution routing

> Add as many models and replicas as you want — Conduit will calculate model constraints and throw an error if something doesn't fit.

---

## 🚀 Example: Batched Inference Workload (with comments)

```python
from dataclasses import dataclass
from typing import List, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from conduit.conduit_types import GPUS
from conduit.runtime import LMLiteBlock
from conduit import ModelConfig, ComputeProvider


def test_batch_processing_with_mdl(
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
) -> None:
    """
    Runs a simulated batch processing workload with LMLite using Conduit.
    Shows how to:
    - configure models + GPUs
    - spin up an LMLiteBlock
    - send 100 concurrent requests
    """

    # LMLite is the runtime block that actually runs the models.
    # Here we define a single model configuration, but you can pass many.
    block = LMLiteBlock(
        models=[
            # Add as many models as you want! Conduit will calculate model and GPU
            # requirements and throw an error if something doesn't fit.
            ModelConfig(
                model_id,  # HF model id
                # Max total tokens per request (input + output).
                max_model_len=1400,
                # Max number of concurrent requests LMLite will batch together.
                max_model_concurrency=50,
                # How long to wait (in ms) before executing a batch, even if not full.
                model_batch_execute_timeout_ms=1000,
            )
        ],
        # Choose compute provider. Currently only RUNPOD is implemented.
        compute_provider=ComputeProvider.RUNPOD,
        # You can pick a specific GPU here. If omitted, Conduit can choose the
        # smallest GPU that still satisfies your model VRAM requirements.
        gpu=GPUS.NVIDIA_L4,
        # Number of replicas of this runtime. Requests are round-robin load balanced.
        replicas=2,
    )

    # IMPORTANT: call after all blocks are defined.
    # It garbage collects any old instances after a config update so you don't
    # leak stale runtimes across restarts.
    LMLiteBlock.gc()

    # Wait until healthcheck passes for each node.
    # Conduit exposes a `.ready` flag so you don't send traffic too early.
    while True:
        print("waiting until healthcheck passes for each node...")
        time.sleep(5)
        if block.ready:  # LMLite is ready when the container healthcheck passes
            break

    # --- Typed IO for your model ---

    @dataclass
    class GreetingInput:
        # Input fields that will be turned into a prompt / structured input.
        name: str
        mood: Optional[str] = None

    @dataclass
    class GreetingOutput:
        # Expected model output schema. Conduit / LMLite will validate against this.
        greeting: str
        suggestions: List[str]

    # Conduit compiles your dataclasses into type hints for the LLM.
    # This is the strongly typed input we'll reuse for all requests.
    inp = GreetingInput(name="Alice", mood="excited")

    # Example for OpenAI-style messages (commented out but supported).
    # You can either send `messages` or typed input/output.
    #
    # result = block(
    #     model_id="Qwen/Qwen3-4B-Thinking-2507",
    #     messages=[{"role": "user", "content": "hi there"}],
    #     system_message="You are a greeter",
    # )

    # Simulate 100 simultaneous client-side requests to this runtime.
    NUM_REQUESTS = 100
    # Parallelism level on the client side (ThreadPoolExecutor).
    MAX_WORKERS = 100

    start = time.time()

    # Spin up a pool of workers and call the block concurrently.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # For each request we submit a call to the block.
        futures = {
            executor.submit(
                block,
                model_id=model_id,   # which logical model to use (must match config above)
                messages=None,       # using typed IO instead of chat messages here
                input=inp,           # typed input dataclass instance
                output=GreetingOutput,  # typed output schema
            ): i
            for i in range(NUM_REQUESTS)
        }

        # As each future completes, print its result and latency.
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = f"ERROR: {e}"

            elapsed = time.time() - start
            print(f"[{idx:03}] completed at {elapsed:6.2f}s → {res}")

    # Clean up: destroy the runtime and its backing resources.
    # This clears the instance and any internal DB references.
    block.delete()

```

## ✅ TODOs

- [ ] Support **vLLM** as a runtime (e.g. `VllmBlock` alongside `LMLiteBlock`)
- [ ] Add a **local compute provider** for self-hosted/on-device GPU execution
- [ ] Expand supported compute providers:
  - [ ] **RunPod** (existing)
  - [ ] **AWS** (EKS / ECS / EC2)
  - [ ] **GCP** (GKE / Vertex / Compute Engine)
  - [ ] **Azure** (AKS / Azure ML)
  - [ ] **Lambda Labs**
  - [ ] **Modal**
  - [ ] **CoreWeave**
  - [ ] **Local / Bare Metal** clusters
