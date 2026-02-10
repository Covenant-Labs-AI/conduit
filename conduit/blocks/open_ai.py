import os
from typing import List, Type, Any, overload, TypeVar

from .block_types import Block
from conduit.conduit_http import OpenAIMessage, inf_open_ai_compat
from conduit.mdl import build_mdl_system_prompt
from conduit.utils import dataclass_to_dict
from conduit.utils.deployment.vram import parse_llm_json

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class OpenAICompatableRuntimeBlock(Block[TIn, TOut]):
    def __init__(
        self,
        *,
        host: str = "api.openai.com",
        port: int | None = 443,
        scheme: str = "https",
        api_key_env: str = "OPENAI_API_KEY",
        require_api_key: bool = True,
    ) -> None:
        if require_api_key and not os.getenv(api_key_env):
            raise RuntimeError(
                f"Missing API key: set {api_key_env} in the environment."
            )
        self.api_key_env = api_key_env
        self.host = host
        self.port = port
        self.scheme = scheme

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
    ) -> Any:
        if messages and input is None and output is None:
            return inf_open_ai_compat(
                self.host, self.port, model_id, messages, guidance, scheme=self.scheme
            )

        if (input is not None and output is not None) and not messages:
            system_prompt = build_mdl_system_prompt(guidance or "", input, output)
            data_input: List[OpenAIMessage] = [
                {"role": "user", "content": str(dataclass_to_dict(input))}
            ]
            json_response = inf_open_ai_compat(
                self.host,
                self.port,
                model_id,
                data_input,
                system_prompt,
                scheme=self.scheme,
                api_key=os.getenv(self.api_key_env),
            )
            return parse_llm_json(json_response, output)

        if (input is not None) ^ (output is not None):
            raise ValueError("Both `input` and `output` must be provided together.")

        raise ValueError("Provide either `messages` or (`input`, `output`).")
