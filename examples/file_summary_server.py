from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Mapping, Optional

from conduit.blocks import FastAPIServerBlock, TypedRouteSpec, FastAPIServerConfig
from conduit.runtime import LMLiteBlock
from conduit.conduit_types import ComputeProvider, LmLiteModelConfig
from conduit.utils.deployment import DeploymentConstraint


@dataclass
class GetFileQuery:
    relpath: str


@dataclass
class FileResponseData:
    path: Path | None = None
    content: bytes | None = None

    media_type: str = "application/octet-stream"
    filename: str | None = None
    headers: Mapping[str, str] | None = None
    status_code: int = 200


@dataclass
class ExplainFileQuery:
    relpath: str
    max_chars: int = 25_000


@dataclass
class FileSnapshot:
    path: str
    size_bytes: int
    truncated: bool
    raw_text: str


@dataclass
class FileInsight:
    kind: Literal["log", "config", "source", "text", "binary", "unknown"]
    summary: str
    key_points: List[str]
    warnings: List[str]


@dataclass
class ExplainFileResponse:
    relpath: str
    resolved_path: str
    size_bytes: int
    truncated: bool
    insight: FileInsight


STATIC_ROOT = Path("./").resolve()

model_id = "Qwen/Qwen3-4B-Instruct-2507-FP8"
lm = LMLiteBlock(
    models=[
        LmLiteModelConfig(
            model_id,
            max_model_len=50_000,
            max_model_concurrency=1,
        )
    ],
    constraints=[DeploymentConstraint.ENTERPRISE],
    compute_provider=ComputeProvider.RUNPOD,
)


def _safe_resolve(relpath: str) -> Path:
    # Prevent absolute paths and weird Windows drive forms
    rel = relpath.lstrip("/\\")
    p = (STATIC_ROOT / rel).resolve()

    # Allow STATIC_ROOT itself? no (must be a file), but allow immediate children and deeper
    if p != STATIC_ROOT and STATIC_ROOT not in p.parents:
        raise RuntimeError("Access denied (path escape)")

    if not p.is_file():
        raise RuntimeError("File not found")

    return p


def _guess_media_type(p: Path) -> str:
    # minimal; keep predictable and avoid extra deps
    ext = p.suffix.lower()
    if ext in (".txt", ".log", ".md"):
        return "text/plain"
    if ext in (".json",):
        return "application/json"
    if ext in (".html", ".htm"):
        return "text/html"
    if ext in (".csv",):
        return "text/csv"
    return "application/octet-stream"


def read_file_text_snapshot(p: Path, max_chars: int) -> FileSnapshot:
    size = p.stat().st_size

    with p.open("rb") as f:
        head = f.read(4096)

    if b"\x00" in head:
        return FileSnapshot(
            path=str(p),
            size_bytes=size,
            truncated=True,
            raw_text="",
        )

    raw_text = p.read_bytes().decode("utf-8", errors="replace")
    max_chars = max(1_000, min(int(max_chars), 200_000))
    truncated = len(raw_text) > max_chars
    if truncated:
        raw_text = raw_text[:max_chars]

    return FileSnapshot(
        path=str(p),
        size_bytes=size,
        truncated=truncated,
        raw_text=raw_text,
    )


def summarize_snapshot(snapshot: FileSnapshot) -> FileInsight:
    if snapshot.raw_text == "" and snapshot.size_bytes > 0:
        return FileInsight(
            kind="binary",
            summary="This appears to be a binary/non-text file; serving raw download only.",
            key_points=[],
            warnings=["Binary file detected (NUL bytes)."],
        )

    guidance = """
You are analyzing the contents of a file loaded from disk.

Rules:
- Classify the file into one of: log, config, source, text, binary, unknown.
- Do NOT restate the file contents.
- Produce a concise human-readable summary.
- `key_points` should be 3–7 bullets capturing what matters.
- `warnings` should include risks, secrets, misconfigurations, or anomalies.
- If the input is truncated, mention that briefly.
- Do not invent facts not present in the file.
""".strip()

    lm.ready

    out = lm(
        model_id=model_id,
        input=snapshot,
        output=FileInsight,
        guidance=guidance,
    )

    if snapshot.truncated:
        # ensure truncation is explicit even if model forgets
        if "Input was truncated for analysis." not in out.warnings:
            out.warnings = ["Input was truncated for analysis."] + list(out.warnings)

    return out


def get_file_handler(q: GetFileQuery) -> FileResponseData:
    p = _safe_resolve(q.relpath)
    return FileResponseData(
        path=p,
        filename=p.name,
        media_type=_guess_media_type(p),
        status_code=200,
    )


def explain_file_handler(q: ExplainFileQuery) -> ExplainFileResponse:
    p = _safe_resolve(q.relpath)
    snap = read_file_text_snapshot(p, q.max_chars)
    insight = summarize_snapshot(snap)
    return ExplainFileResponse(
        relpath=q.relpath,
        resolved_path=snap.path,
        size_bytes=snap.size_bytes,
        truncated=snap.truncated,
        insight=insight,
    )


server = FastAPIServerBlock()

op = server(
    FastAPIServerConfig(
        routes=[
            # 1) Raw file download
            TypedRouteSpec(
                method="GET",
                path="/file/raw",
                input=GetFileQuery,
                output=FileResponseData,
                handler=get_file_handler,
                name="get_file_raw",
            ),
            # 2) JSON insight
            TypedRouteSpec(
                method="GET",
                path="/file/explain",
                input=ExplainFileQuery,
                output=ExplainFileResponse,
                handler=explain_file_handler,
                name="explain_file",
            ),
        ],
        host="127.0.0.1",
        port=8080,
        log_level="info",
        ready_timeout_s=10.0,
    )
)

if not op.success:
    raise RuntimeError(op.reason or "Server failed to start")

print(f"Server ready: {op.base_url}")
print(f"Raw:     {op.base_url}/file/raw?relpath=ai.txt")
print(f"Explain:  {op.base_url}/file/explain?relpath=ai.txt&max_chars=5000")

# Keep alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    lm.delete()
    server.stop()
