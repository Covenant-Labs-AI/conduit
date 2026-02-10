from dataclasses import dataclass
from conduit.compute_provider.runpod.runpod_types import GPUS
from conduit.runtime import VLLMBlock

from conduit.conduit_types import ComputeProvider, VLLMModelConfig
from conduit.utils.deployment import DeploymentConstraint

vllm_block = VLLMBlock(
    models=[VLLMModelConfig(id="Qwen/Qwen3-4B-Instruct-2507-FP8", max_model_len=1024)],
    compute_provider=ComputeProvider.RUNPOD,
    compute_provider_config_overrides={"countryCodes": ["US"]},
    constraints=[
        DeploymentConstraint.ENTERPRISE,
        DeploymentConstraint.HIGH_BANDWIDTH_INTERCONNECT,
    ],
    gpu=GPUS.L40,
)
PROMPT = """
Felis catus is your taxonomic nomenclature,
An endothermic quadruped, carnivorous by nature;
Your visual, olfactory, and auditory senses
Contribute to your hunting skills and natural defenses.
"""


@dataclass
class Poem:
    start: str = PROMPT


@dataclass
class PoemEnd:
    end: str


if vllm_block.ready:
    out = vllm_block(
        model_id="Qwen/Qwen3-4B-Instruct-2507-FP8",
        input=Poem(),
        output=PoemEnd,
        guidance="Complete this poem",
    )

    print(out)
