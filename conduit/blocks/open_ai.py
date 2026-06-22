import os
from typing import Any, Callable, List, Type, TypeVar, cast, overload

from conduit.conduit_types import LLMRawTrace
from openai import APIError, BadRequestError, NotFoundError, OpenAI

from conduit.conduit_http import OpenAIMessage, inf_open_ai_compat
from conduit.mdl import build_mdl_system_prompt
from conduit.utils import dataclass_to_dict
from conduit.utils.deployment.vram import parse_llm_json

from .block_types import Block

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class OpenAICompatibleRuntimeBlock(Block[TIn, TOut]):
    def __init__(
        self,
        *,
        api_key: str,
        host: str = "api.openai.com",
        port: int | None = 443,
        scheme: str = "https",
        require_api_key: bool = True,
        api_mode: str = "auto",  # "auto" | "responses" | "chat_completions"
    ) -> None:

        if api_mode not in {"auto", "responses", "chat_completions"}:
            raise ValueError(
                "api_mode must be one of: 'auto', 'responses', 'chat_completions'"
            )
        self.api_key = api_key
        self.host = host
        self.port = port
        self.scheme = scheme
        self.api_mode = api_mode

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
        output: Type[TOut],
    ) -> TOut: ...

    def __call__(
        self,
        model_id: str,
        messages: List[OpenAIMessage] | None = None,
        guidance: str | None = None,
        *,
        input: Any = None,
        output: Type[TOut] | None = None,
        raw_trace_hook: Callable[[LLMRawTrace], None] | None = None,
    ) -> Any:
        if messages and input is None and output is None:
            raw_input = str(messages)

            raw_output = inf_open_ai_compat(
                self.host,
                self.port,
                model_id,
                messages,
                system_message=guidance,
                scheme=self.scheme,
                api_key=self.api_key,
                api_mode=self.api_mode,
            )

            if raw_trace_hook is not None:
                raw_trace_hook(
                    LLMRawTrace(
                        raw_input=raw_input,
                        raw_output=raw_output,
                    )
                )

            return raw_output

        if (input is not None and output is not None) and not messages:
            system_prompt = build_mdl_system_prompt(guidance or "", input, output)
            data_input: List[OpenAIMessage] = [
                {"role": "user", "content": str(dataclass_to_dict(input))}
            ]

            raw_input = f"SYSTEM:\n{system_prompt}\n\nMESSAGES:\n{data_input}"

            raw_output = inf_open_ai_compat(
                self.host,
                self.port,
                model_id,
                data_input,
                system_message=system_prompt,
                scheme=self.scheme,
                api_key=os.getenv(self.api_key_env),
                api_mode=self.api_mode,
            )

            if raw_trace_hook is not None:
                raw_trace_hook(
                    LLMRawTrace(
                        raw_input=raw_input,
                        raw_output=raw_output,
                    )
                )

            return parse_llm_json(raw_output, output)

        if (input is not None) ^ (output is not None):
            raise ValueError("Both `input` and `output` must be provided together.")

        raise ValueError("Provide either `messages` or (`input`, `output`).")

    @property
    def ready(self) -> bool:
        return True
