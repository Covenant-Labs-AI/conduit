from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict, cast
from urllib.parse import urlparse

import requests
from openai import APIError, BadRequestError, NotFoundError, OpenAI


class OpenAIMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class LLMCompletion:
    content: str | None
    reasoning: str | None = None
    finish_reason: str | None = None

    def require_content(self) -> str:
        if isinstance(self.content, str) and self.content.strip():
            return self.content

        details: list[str] = []

        if self.finish_reason:
            details.append(f"finish_reason={self.finish_reason!r}")

        if self.reasoning:
            preview = " ".join(self.reasoning.split())[:500]
            details.append(f"reasoning_preview={preview!r}")

        suffix = f" ({', '.join(details)})" if details else ""
        raise RuntimeError(f"Model returned no final message content{suffix}.")


@dataclass
class EmbeddingItem:
    index: int
    embedding: list[float] | str


@dataclass
class EmbeddingResponse:
    model: str
    data: list[EmbeddingItem]
    prompt_tokens: int
    total_tokens: int


@dataclass
class RerankResult:
    index: int
    relevance_score: float
    document: str | None = None


@dataclass
class RerankResponse:
    id: str
    model: str
    results: list[RerankResult]
    prompt_tokens: int
    total_tokens: int


def _parse_host_port_scheme(
    host: str,
    port: int | None = None,
    scheme: str = "http",
) -> tuple[str, int, str]:
    parsed = urlparse(host)

    if parsed.scheme:
        scheme = parsed.scheme
        host = parsed.hostname or ""
        port = parsed.port or port
    elif ":" in host and host.count(":") == 1 and host.split(":")[1].isdigit():
        host, raw_port = host.split(":")
        port = int(raw_port)

    if port is None:
        port = 80 if scheme == "http" else 443

    return host, port, scheme


def check_endpoint(
    host: str,
    port: int | None = None,
    path: str = "/health",
    scheme: str = "http",
) -> bool:
    host, port, scheme = _parse_host_port_scheme(host, port, scheme)
    url = f"{scheme}://{host}:{port}{path}"

    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def healthcheck(host: str, port: int) -> bool:
    return check_endpoint(host, port, "/health")


def metrics(host: str, port: int) -> bool:
    return check_endpoint(host, port, "/metrics")


def inf_embedding(
    host: str,
    port: int | None,
    model_id: str,
    input: str | List[str],
    *,
    encoding_format: Literal["float", "base64"] = "float",
    dimensions: int | None = None,
    scheme: str = "http",
    api_key: str | None = "default-placeholder",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    host, port, scheme = _parse_host_port_scheme(host, port, scheme)
    url = f"{scheme}://{host}:{port}/v1/embeddings"

    payload: Dict[str, Any] = {
        "model": model_id,
        "input": input,
        "encoding_format": encoding_format,
    }

    if dimensions is not None:
        payload["dimensions"] = dimensions

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    return cast(Dict[str, Any], response.json())


def inf_rerank(
    host: str,
    port: int | None,
    model_id: str,
    query: str,
    documents: List[str],
    *,
    top_n: int | None = None,
    return_documents: bool = False,
    scheme: str = "http",
    api_key: str | None = "default-placeholder",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    if not documents:
        raise ValueError("documents must be a non-empty list of strings")

    host, port, scheme = _parse_host_port_scheme(host, port, scheme)
    url = f"{scheme}://{host}:{port}/v1/rerank"

    payload: Dict[str, Any] = {
        "model": model_id,
        "query": query,
        "documents": documents,
        "return_documents": return_documents,
    }

    if top_n is not None:
        payload["top_n"] = top_n

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    return cast(Dict[str, Any], response.json())


def inf_open_ai_compat(
    host: str,
    port: int | None,
    model_id: str,
    messages: List[OpenAIMessage],
    system_message: str | None = None,
    scheme: str = "http",
    api_key: str | None = "default-placeholder",
    api_mode: str = "auto",
    max_tokens: int | None = None,
) -> str:
    completion = inf_open_ai_compat_completion(
        host=host,
        port=port,
        model_id=model_id,
        messages=messages,
        system_message=system_message,
        scheme=scheme,
        api_key=api_key,
        api_mode=api_mode,
        max_tokens=max_tokens,
    )

    return completion.require_content()


def inf_open_ai_compat_completion(
    host: str,
    port: int | None,
    model_id: str,
    messages: List[OpenAIMessage],
    system_message: str | None = None,
    scheme: str = "http",
    api_key: str | None = "default-placeholder",
    api_mode: str = "auto",
    max_tokens: int | None = None,
) -> LLMCompletion:
    _validate_messages(messages)

    host, port, scheme = _parse_host_port_scheme(host, port, scheme)
    base_url = _build_base_url(host, port, scheme)

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    final_messages = _with_system_message(messages, system_message)

    if api_mode == "responses":
        return _call_responses_api(client, model_id, final_messages)

    if api_mode == "chat_completions":
        return _call_chat_completions_api(
            client, model_id, final_messages, max_tokens=max_tokens
        )

    try:
        return _call_responses_api(client, model_id, final_messages)
    except (NotFoundError, BadRequestError, APIError) as exc:
        if _should_fallback_to_chat(exc):
            return _call_chat_completions_api(
                client, model_id, final_messages, max_tokens=max_tokens
            )
        raise


def _validate_messages(messages: List[OpenAIMessage]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of {role, content} dicts")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"Invalid message format: {message}")

        if "role" not in message or "content" not in message:
            raise ValueError(f"Invalid message format: {message}")


def _with_system_message(
    messages: List[OpenAIMessage],
    system_message: str | None,
) -> List[OpenAIMessage]:
    final_messages: List[OpenAIMessage] = []

    if system_message:
        final_messages.append(
            {
                "role": "system",
                "content": system_message,
            }
        )

    final_messages.extend(messages)

    return final_messages


def _build_base_url(host: str, port: int | None, scheme: str) -> str:
    if port is None:
        return f"{scheme}://{host}/v1"

    return f"{scheme}://{host}:{port}/v1"


def _call_responses_api(
    client: OpenAI,
    model_id: str,
    messages: List[OpenAIMessage],
) -> LLMCompletion:
    response = client.responses.create(
        model=model_id,
        input=messages,
    )

    content = getattr(response, "output_text", None)

    if not isinstance(content, str) or not content.strip():
        content = _extract_text_from_responses_output(response)

    reasoning = _extract_reasoning_from_responses_output(response)

    content, tagged_reasoning = _split_leading_think_block(content)
    reasoning = _join_text(reasoning, tagged_reasoning)

    return LLMCompletion(
        content=content,
        reasoning=reasoning,
        finish_reason=_get_model_field(response, "status"),
    )


def _call_chat_completions_api(
    client: OpenAI,
    model_id: str,
    messages: List[OpenAIMessage],
    *,
    max_tokens: int | None = None,
) -> LLMCompletion:

    request: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }

    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    response = client.chat.completions.create(**request)

    if not response.choices:
        raise RuntimeError("Chat Completions API returned no choices.")

    choice = response.choices[0]
    message = choice.message

    content = _extract_message_content(message)
    reasoning = _extract_message_reasoning(message)

    content, tagged_reasoning = _split_leading_think_block(content)
    reasoning = _join_text(reasoning, tagged_reasoning)

    return LLMCompletion(
        content=content,
        reasoning=reasoning,
        finish_reason=getattr(choice, "finish_reason", None),
    )


def _extract_message_content(message: Any) -> str | None:
    return _coerce_text(_get_model_field(message, "content"))


def _extract_message_reasoning(message: Any) -> str | None:
    parts: list[str] = []

    for field_name in (
        "reasoning_content",
        "reasoning",
        "thinking",
    ):
        value = _coerce_text(_get_model_field(message, field_name))

        if value and value not in parts:
            parts.append(value)

    return "\n".join(parts) if parts else None


def _extract_text_from_responses_output(response: Any) -> str | None:
    output = getattr(response, "output", None)

    if not isinstance(output, list):
        return None

    parts: list[str] = []

    for item in output:
        if _get_model_field(item, "type") != "message":
            continue

        content = _get_model_field(item, "content")

        if not isinstance(content, list):
            continue

        for chunk in content:
            if _get_model_field(chunk, "type") != "output_text":
                continue

            text = _coerce_text(_get_model_field(chunk, "text"))

            if text:
                parts.append(text)

    return "".join(parts) if parts else None


def _extract_reasoning_from_responses_output(response: Any) -> str | None:
    output = getattr(response, "output", None)

    if not isinstance(output, list):
        return None

    parts: list[str] = []

    for item in output:
        if _get_model_field(item, "type") != "reasoning":
            continue

        text = _coerce_text(_get_model_field(item, "summary"))

        if not text:
            text = _coerce_text(_get_model_field(item, "content"))

        if text:
            parts.append(text)

    return "\n".join(parts) if parts else None


def _get_model_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)

    field_value = getattr(value, field_name, None)

    if field_value is not None:
        return field_value

    model_extra = getattr(value, "model_extra", None)

    if isinstance(model_extra, dict):
        return model_extra.get(field_name)

    return None


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return None

    if isinstance(value, list):
        parts = [_coerce_text(item) for item in value]
        return _join_text(*parts)

    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "reasoning_content",
            "reasoning",
            "thinking",
            "summary",
        ):
            text = _coerce_text(value.get(key))
            if text:
                return text

        return None

    for field_name in (
        "text",
        "content",
        "reasoning_content",
        "reasoning",
        "thinking",
        "summary",
    ):
        field_value = _get_model_field(value, field_name)

        if field_value is None or field_value is value:
            continue

        text = _coerce_text(field_value)

        if text:
            return text

    return None


def _split_leading_think_block(
    content: str | None,
) -> tuple[str | None, str | None]:
    if not isinstance(content, str):
        return None, None

    stripped = content.lstrip()

    if not stripped.startswith("<think>"):
        return content, None

    thought_start = len("<think>")
    thought_end = stripped.find("</think>", thought_start)

    if thought_end == -1:
        reasoning = stripped[thought_start:].strip()
        return None, reasoning or None

    reasoning = stripped[thought_start:thought_end].strip()
    final_content = stripped[thought_end + len("</think>") :].strip()

    return (
        final_content or None,
        reasoning or None,
    )


def _join_text(*values: str | None) -> str | None:
    parts = [
        value.strip() for value in values if isinstance(value, str) and value.strip()
    ]

    if not parts:
        return None

    return "\n".join(parts)


def _should_fallback_to_chat(exc: Exception) -> bool:
    message = str(exc).lower()

    responses_not_supported_signals = (
        "responses",
        "/v1/responses",
        "only supported in v1/chat/completions",
        "unsupported",
        "not found",
    )

    return any(signal in message for signal in responses_not_supported_signals)
