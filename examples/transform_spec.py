from dataclasses import dataclass
from typing import List

from conduit.transform import TransformManager


@dataclass
class SomeInput:
    text: str


@dataclass
class SomeOutput:
    title: str
    bullets: List[str]


manager = TransformManager.from_json_file("./examples/specs/transform_spec.json")

summarize = manager.transform("summarize_v1")
summarize.ready

out = summarize(
    input=SomeInput(text="Conduit provides type-safe, provider-agnostic AI pipelines."),
    output=SomeOutput,
    guidance="Extract a short title and 3 concise bullets.",
)

print(out)
