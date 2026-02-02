from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from conduit import NoOp
from conduit.blocks import (
    SystemCommandBlock,
    FastAPIServerBlock,
    TypedRouteSpec,
    FastAPIServerConfig,
)
from conduit.runtime import LMLiteBlock
from conduit.conduit_types import ComputeProvider, LmLiteModelConfig
from conduit.utils.deployment import DeploymentConstraint


@dataclass
class Command:
    shell_command: str


@dataclass
class TopSnapshot:
    host: str
    captured_at_unix_s: int
    raw_top: str


@dataclass
class ProcRow:
    pid: int
    user: str
    cpu_percent: float
    mem_percent: float
    command: str


@dataclass
class TopSummary:
    headline: str
    load_avg: Optional[str]
    top_cpu: List[ProcRow]
    top_mem: List[ProcRow]
    notes: List[str]


@dataclass
class QueryParams:
    """
    GET query params:
      ?k=5               how many processes to return (default 5)
      &sort=cpu|mem      which list to emphasize in headline/notes
    """

    k: int = 5
    sort: str = "cpu"


@dataclass
class SummaryResponse:
    summary: TopSummary


cmd = SystemCommandBlock(Command, timeout_seconds=5)

model_id = "Qwen/Qwen3-4B-Instruct-2507-FP8"
lm = LMLiteBlock(
    models=[
        LmLiteModelConfig(
            model_id,
            max_model_len=75_000,
            max_model_concurrency=1,
        )
    ],
    constraints=[DeploymentConstraint.ENTERPRISE],
    compute_provider=ComputeProvider.RUNPOD,
)


def read_top() -> TopSnapshot:
    """
    Use top in batch mode for a single snapshot.
    -b : batch mode
    -n 1 : 1 iteration
    -w 512 : widen output (helps avoid truncation)
    """
    op = cmd(Command(shell_command="top -b -n 1 -w 512"))
    if not op.success:
        raise RuntimeError(op.reason or "Failed to run top")
    raw = op.stdout or ""
    return TopSnapshot(
        host="localhost", captured_at_unix_s=int(time.time()), raw_top=raw
    )


def summarize_top(snapshot: TopSnapshot, k: int, sort: str) -> TopSummary:
    """
    Typed structured output via MDL (dataclass -> dataclass).
    """
    guidance = f"""
You are given the output of Linux `top` as text in `raw_top`.
Produce a concise structured summary.

Rules:
- Parse load averages if present.
- Return exactly {k} items in `top_cpu` and {k} items in `top_mem` when possible.
- Each ProcRow must include: pid, user, cpu_percent, mem_percent, command.
- `command` should be the COMMAND column (trim long commands).
- `headline` should be one sentence describing overall status.
- `notes` should be 2–6 short bullets (no markdown), include anomalies if any.
- If you cannot parse a field, make a best effort and keep types valid.
- Do not invent processes that are not in the input.
Sort emphasis: {sort}
""".strip()
    lm.ready

    out = lm(
        model_id=model_id,
        input=snapshot,
        output=TopSummary,
        guidance=guidance,
    )
    return out


# -------------------------
# Server handler
# -------------------------


def get_summary(q: QueryParams) -> SummaryResponse:
    # Clamp k to keep output + parsing stable
    k = max(1, min(int(q.k), 20))
    sort = (q.sort or "cpu").lower()
    if sort not in ("cpu", "mem"):
        sort = "cpu"

    snapshot = read_top()
    summary = summarize_top(snapshot, k=k, sort=sort)
    return SummaryResponse(summary=summary)


server = FastAPIServerBlock()

op = server(
    FastAPIServerConfig(
        routes=[
            TypedRouteSpec(
                method="GET",
                path="/summary",
                input=QueryParams,
                output=SummaryResponse,
                handler=get_summary,
                name="get_top_summary",
            )
        ],
        host="127.0.0.1",
        port=9191,
        log_level="info",
        ready_timeout_s=10.0,
    )
)

if not op.success:
    raise RuntimeError(op.reason or "Server failed to start")

print(f"Server ready: {op.base_url}")
print(f"Try: {op.base_url}/summary?k=5&sort=cpu")

# IMPORTANT: keep the process alive (server runs in a daemon thread)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    lm.delete()
    server.stop()
