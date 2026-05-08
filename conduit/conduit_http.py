import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict, cast
from urllib.parse import urlparse

import requests
from openai import APIError, BadRequestError, NotFoundError, OpenAI


class OpenAIMessage(TypedDict):
    role: str
    content: str


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
    host: str, port: int | None = None, path: str = "/health", scheme: str = "http"
):
    host, port, scheme = _parse_host_port_scheme(host, port, scheme)
    url = f"{scheme}://{host}:{port}{path}"

    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def healthcheck(host: str, port: int):
    return check_endpoint(host, port, "/health")


def metrics(host: str, port: int):
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
    api_mode: str = "auto",  # "auto" | "responses" | "chat_completions"
) -> str:
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
        return _call_chat_completions_api(client, model_id, final_messages)

    # auto mode: prefer Responses API, then fall back to Chat Completions
    try:
        return _call_responses_api(client, model_id, final_messages)
    except (NotFoundError, BadRequestError, APIError) as exc:
        if _should_fallback_to_chat(exc):
            return _call_chat_completions_api(client, model_id, final_messages)
        raise


def _validate_messages(messages: List[OpenAIMessage]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of {role, content} dicts")

    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"Invalid message format: {m}")


def _with_system_message(
    messages: List[OpenAIMessage],
    system_message: str | None,
) -> List[OpenAIMessage]:
    final_messages: List[OpenAIMessage] = []
    if system_message:
        final_messages.append({"role": "system", "content": system_message})
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
) -> str:
    response = client.responses.create(
        model=model_id,
        input=messages,
    )
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text

    # More defensive extraction in case output_text is empty or unavailable.
    extracted = _extract_text_from_responses_output(response)
    if extracted:
        return extracted

    raise RuntimeError("Responses API returned no text output.")


def _call_chat_completions_api(
    client: OpenAI,
    model_id: str,
    messages: List[OpenAIMessage],
) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    if not response.choices:
        raise RuntimeError("Chat Completions API returned no choices.")

    message = response.choices[0].message
    content = message.content

    if isinstance(content, str):
        return content

    if content is None:
        raise RuntimeError("Chat Completions API returned no message content.")

    # Defensive handling if SDK/provider returns structured content.
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)

    raise RuntimeError("Unable to extract text from Chat Completions response.")


def _extract_text_from_responses_output(response: Any) -> str:
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return ""

    parts: list[str] = []

    for item in output:
        item_type = getattr(item, "type", None)
        if item_type != "message":
            continue

        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue

        for chunk in content:
            chunk_type = getattr(chunk, "type", None)
            if chunk_type == "output_text":
                text = getattr(chunk, "text", None)
                if isinstance(text, str):
                    parts.append(text)

    return "".join(parts)


def _should_fallback_to_chat(exc: Exception) -> bool:
    msg = str(exc).lower()

    responses_not_supported_signals = (
        "responses",
        "/v1/responses",
        "only supported in v1/chat/completions",
        "unsupported",
        "not found",
    )

    return any(signal in msg for signal in responses_not_supported_signals)
