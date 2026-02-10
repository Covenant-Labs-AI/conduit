import os
from dataclasses import dataclass

from conduit.blocks import OpenAICompatableRuntimeBlock


@dataclass
class SummarizeIn:
    text: str


@dataclass
class SummarizeOut:
    summary: str
    bullets: list[str]


client = OpenAICompatableRuntimeBlock()

result = client(
    "gpt-4.1-mini",
    input=SummarizeIn(
        text="Conduit provisions GPU containers and exposes an OpenAI-compatible API."
    ),
    output=SummarizeOut,
    guidance="write marketing speak",
)

print(result)  # SummarizeOut(summary='...', bullets=[...])
print(result.summary)
print(result.bullets)
