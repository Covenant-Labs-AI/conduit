import base64
import mimetypes
from pathlib import Path
from typing import Any, Callable, List, Type, TypeVar, cast, overload

from conduit.blocks.block_types import Block
from conduit.conduit_http import (
    LLMCompletion,
    OpenAIMessage,
    inf_open_ai_compat_completion,
)
from conduit.conduit_types import LLMRawTrace
from conduit.mdl import build_mdl_system_prompt
from conduit.utils import dataclass_to_dict
from conduit.utils.deployment.vram import parse_llm_json


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
        api_mode: str = "auto",
        max_tokens: int | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if api_mode not in {"auto", "responses", "chat_completions"}:
            raise ValueError(
                "api_mode must be one of: 'auto', 'responses', 'chat_completions'"
            )

        self.max_tokens = max_tokens
        self.request_kwargs = request_kwargs or {}
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
        image_urls: list[str] | None = ...,
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
        image_urls: list[str] | None = ...,
    ) -> TOut: ...

    def __call__(
        self,
        model_id: str,
        messages: List[OpenAIMessage] | None = None,
        guidance: str | None = None,
        *,
        input: Any = None,
        output: Type[TOut] | None = None,
        image_urls: list[str] | None = None,
        raw_trace_hook: Callable[[LLMRawTrace], None] | None = None,
    ) -> Any:
        if messages and input is None and output is None:
            raw_input = _messages_trace_text(messages)

            completion = self._completion(
                model_id=model_id,
                messages=messages,
                system_message=guidance,
            )

            raw_output = completion.require_content()

            self._emit_trace(
                raw_trace_hook=raw_trace_hook,
                raw_input=raw_input,
                completion=completion,
            )

            return raw_output

        if input is not None and output is not None and not messages:
            system_prompt = build_mdl_system_prompt(
                guidance or "",
                input,
                output,
            )

            data_input = _build_input_messages(
                input=input,
                image_urls=image_urls,
            )

            raw_input = (
                f"SYSTEM:\n{system_prompt}\n\n"
                f"MESSAGES:\n{_messages_trace_text(data_input)}"
            )

            completion = self._completion(
                model_id=model_id,
                messages=data_input,
                system_message=system_prompt,
            )

            raw_output = completion.require_content()

            self._emit_trace(
                raw_trace_hook=raw_trace_hook,
                raw_input=raw_input,
                completion=completion,
            )

            return parse_llm_json(raw_output, output)

        if (input is not None) ^ (output is not None):
            raise ValueError("Both `input` and `output` must be provided together.")

        raise ValueError("Provide either `messages` or (`input`, `output`).")

    def _completion(
        self,
        *,
        model_id: str,
        messages: List[OpenAIMessage],
        system_message: str | None,
    ) -> LLMCompletion:
        return inf_open_ai_compat_completion(
            host=self.host,
            port=self.port,
            model_id=model_id,
            messages=messages,
            system_message=system_message,
            scheme=self.scheme,
            api_key=self.api_key,
            api_mode=self.api_mode,
            max_tokens=self.max_tokens,
            request_kwargs=self.request_kwargs,
        )

    def _emit_trace(
        self,
        *,
        raw_trace_hook: Callable[[LLMRawTrace], None] | None,
        raw_input: str,
        completion: LLMCompletion,
    ) -> None:
        if raw_trace_hook is None:
            return

        raw_trace_hook(
            LLMRawTrace(
                raw_input=raw_input,
                raw_output=_completion_trace_text(completion),
            )
        )

    @property
    def ready(self) -> bool:
        return True


def _build_input_messages(
    *,
    input: Any,
    image_urls: list[str] | None,
) -> List[OpenAIMessage]:
    text = str(dataclass_to_dict(input))

    if not image_urls:
        return [
            {
                "role": "user",
                "content": text,
            }
        ]

    content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {
                "url": _resolve_image_url(image_url),
            },
        }
        for image_url in image_urls
    ]

    content.append(
        {
            "type": "text",
            "text": text,
        }
    )

    return [
        cast(
            OpenAIMessage,
            {
                "role": "user",
                "content": content,
            },
        )
    ]


def _resolve_image_url(image_url: str) -> str:
    if image_url.startswith(("http://", "https://", "data:")):
        return image_url

    return _local_image_to_data_url(image_url)


def _local_image_to_data_url(image_path: str) -> str:
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)

    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"Could not determine image MIME type: {path}")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def _messages_trace_text(
    messages: List[OpenAIMessage],
) -> str:
    """
    Render messages for tracing without storing embedded base64 images.
    """
    sanitized: list[dict[str, Any]] = []

    for message in messages:
        copied = dict(message)
        content = copied.get("content")

        if isinstance(content, list):
            sanitized_content: list[Any] = []

            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_url = part.get("image_url")

                    if isinstance(image_url, dict):
                        url = image_url.get("url")

                        if isinstance(url, str) and url.startswith("data:"):
                            mime_type = url.split(";", 1)[0].removeprefix("data:")

                            sanitized_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (f"<embedded {mime_type} omitted>")
                                    },
                                }
                            )
                            continue

                sanitized_content.append(part)

            copied["content"] = sanitized_content

        sanitized.append(copied)

    return str(sanitized)


def _completion_trace_text(
    completion: LLMCompletion,
) -> str:
    parts: list[str] = []

    if completion.reasoning:
        parts.append(f"<think>\n{completion.reasoning}\n</think>")

    if completion.content:
        parts.append(completion.content)

    return "\n\n".join(parts)
