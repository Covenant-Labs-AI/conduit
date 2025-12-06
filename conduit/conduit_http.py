import requests
from openai import OpenAI
from typing import cast, List, Dict, TypedDict
from urllib.parse import urlparse


class OpenAIMessage(TypedDict):
    role: str
    content: str


def check_endpoint(
    host: str, port: int | None = None, path: str = "/health", scheme: str = "http"
):
    """
    Supports:
    - host="10.0.0.12", port=8000
    - host="example.com", port=443, scheme="https"
    - host="https://my-host.com:8000" → auto-detects everything
    """

    # If host is already a full URL, extract components
    parsed = urlparse(host)
    if parsed.scheme:  # user passed full URL like "https://1.2.3.4:8080"
        scheme = parsed.scheme
        host = parsed.hostname
        port = parsed.port or port  # explicit port overrides
    elif ":" in host and host.count(":") == 1 and host.split(":")[1].isdigit():
        # Handle "host:port" format (e.g., "1.2.3.4:8080")
        host, port = host.split(":")
        port = int(port)

    if port is None:
        port = 80 if scheme == "http" else 443  # sensible defaults

    url = f"{scheme}://{host}:{port}{path}"

    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            if "application/json" in response.headers.get("Content-Type", ""):
                return response.json()
            return response.text
        return False
    except requests.exceptions.RequestException:
        return False


def healthcheck(host: str, port: int):
    return check_endpoint(host, port, "/health")


def metrics(host: str, port: int):
    return check_endpoint(host, port, "/metrics")


def inf_open_ai_compat(
    host: str,
    port: int | None,
    model_id: str,
    messages: List[OpenAIMessage],
    system_message: str | None = None,
    scheme: str = "http",
) -> str:
    """
    Supports:
    - host="10.0.0.12", port=8000
    - host="example.com", port=443, scheme="https"
    - host="https://my-host.com:8000"
    - host="1.2.3.4:8080", port=None

    Parameters:
    - messages: List of {"role": ..., "content": ...}
    - system_message: Optional string, prepended to messages
    """

    # Validate messages
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of {role, content} dicts")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError(f"Invalid message format: {m}")

    # If host contains scheme/port, extract them
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

    base_url = f"{scheme}://{host}:{port}/v1"

    client = OpenAI(
        base_url=base_url,
        api_key="lm-lite",  # Dummy key for lm-lite
    )

    # Final message list (system first, if given)
    final_messages = []
    if system_message:
        final_messages.append({"role": "system", "content": system_message})
    final_messages.extend(messages)

    response = client.chat.completions.create(
        model=model_id,
        messages=final_messages,
    )

    return cast(str, response.choices[0].message.content)
