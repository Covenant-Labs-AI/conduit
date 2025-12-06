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
    Runs a simulated batch processing workload with LMLite
    """

    block = LMLiteBlock(
        models=[
            # Add as many models as you want! Conduit will calculate model and GPU
            # requirements and throw an error if something doesn't fit.
            ModelConfig(
                model_id,  # HF model id
                max_model_len=1400,  # total tokens per individual request input AND output
                max_model_concurrency=50,  # batch size pool, LMLite queues and processes requests in batches
                model_batch_execute_timeout_ms=1000,  # waits this long before processing batch
            )
        ],
        compute_provider=ComputeProvider.RUNPOD,  # choose compute provider (only RUNPOD atm)
        # choose a specific GPU OR by default Conduit will optimize for VRAM efficiency
        # on smallest possible GPU for your model(s) requirements
        gpu=GPUS.NVIDIA_L4,
        # default = 1. Makes many copies of LMLite; this setup uses a simple
        # round-robin load balancing strategy across replicas.
        replicas=2,
    )

    # IMPORTANT: call after all blocks are defined.
    # It garbage collects any old instances after a config update.
    LMLiteBlock.gc()

    # Wait until healthcheck passes for each node
    while True:
        print("waiting until healthcheck passes for each node...")
        time.sleep(5)
        if block.ready:  # LMLite is ready when the container healthcheck passes
            break

    @dataclass
    class GreetingInput:
        name: str
        mood: Optional[str] = None

    @dataclass
    class GreetingOutput:
        greeting: str
        suggestions: List[str]

    # Conduit compiles your dataclasses into type hints for the LLM
    inp = GreetingInput(name="Alice", mood="excited")

    # Example for OpenAI-style messages:
    # result = block(
    #     model_id="Qwen/Qwen3-4B-Thinking-2507",
    #     messages=[{"role": "user", "content": "hi there"}],
    #     system_message="You are a greeter",
    # )

    # Simulate 100 simultaneous requests
    NUM_REQUESTS = 100
    MAX_WORKERS = 100  # Parallelism level on the client side

    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                block,
                model_id=model_id,
                messages=None,
                input=inp,
                output=GreetingOutput,
            ): i
            for i in range(NUM_REQUESTS)
        }

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = f"ERROR: {e}"

            elapsed = time.time() - start
            print(f"[{idx:03}] completed at {elapsed:6.2f}s → {res}")

    # Obliterates instance and db ref
    block.delete()


if __name__ == "__main__":
    test_batch_processing_with_mdl()
