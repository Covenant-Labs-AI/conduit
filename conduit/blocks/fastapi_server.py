import io
import requests
from dataclasses import dataclass, is_dataclass, asdict
from pathlib import Path
from typing import (
    TypeGuard,
    Type,
    Any,
    Protocol,
    Mapping,
    List,
    Literal,
    runtime_checkable,
)


import threading
import time
from dataclasses import dataclass, is_dataclass, asdict, fields
from typing import List, Literal, Type, Callable, Any, get_type_hints


import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse


from .block_types import Block, I, O

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]


@runtime_checkable
class SupportsFileResponse(Protocol):
    path: Path | None
    content: bytes | None

    media_type: str
    filename: str | None
    headers: Mapping[str, str] | None
    status_code: int


def _is_file_response(x: object) -> TypeGuard[SupportsFileResponse]:
    return (
        hasattr(x, "path")
        and hasattr(x, "content")
        and hasattr(x, "media_type")
        and hasattr(x, "filename")
        and hasattr(x, "headers")
        and hasattr(x, "status_code")
    )


def _coerce_dataclass(cls: Type[I], payload: dict[str, Any]) -> I:
    """
    Create dataclass instance from dict, ignoring unknown fields,
    and applying simple type coercion for primitives.
    """
    if not is_dataclass(cls):
        raise TypeError(f"Input type must be a dataclass type, got {cls}")

    hints = get_type_hints(cls)
    allowed = {f.name for f in fields(cls)}

    init_kwargs: dict[str, Any] = {}
    for k, v in payload.items():
        if k not in allowed:
            continue
        t = hints.get(k)
        # TODO very light coercion (you can expand this later)
        try:
            if t in (int, float, str, bool) and v is not None:
                init_kwargs[k] = t(v)
            else:
                init_kwargs[k] = v
        except Exception:
            init_kwargs[k] = v

    return cls(**init_kwargs)  # type: ignore[arg-type]


async def _parse_request_into(request: Request, input_type: Type[I], method: str) -> I:
    if input_type.__name__ == "NoOp":
        return input_type()  # type: ignore[call-arg]

    # GET -> query params; others -> JSON body (default)
    if method.upper() == "GET":
        payload = dict(request.query_params)
        return _coerce_dataclass(input_type, payload)

    # Non-GET: try JSON body; fall back to empty dict
    try:
        payload = await request.json()
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TypeError("JSON body must be an object")
    except Exception:
        payload = {}

    return _coerce_dataclass(input_type, payload)


@dataclass
class TypedRouteSpec:
    method: HttpMethod
    path: str
    input: Type[I]
    output: Type[O]
    handler: Callable[[I], O]  # can also be async
    name: str | None = None
    status_code: int | None = None


@dataclass
class FastAPIServerConfig:
    routes: List[TypedRouteSpec]
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    ready_timeout_s: float = 10.0


@dataclass
class FastAPIServerOperation:
    success: bool
    base_url: str | None = None
    reason: str | None = None


class FastAPIServerBlock(Block[FastAPIServerConfig, FastAPIServerOperation]):
    """
    Typed FastAPI server:
      HTTP -> (parse) -> Input dataclass -> handler -> Output dataclass -> JSON

    Runs uvicorn in a background thread (same process, non-blocking).
    """

    def __init__(self):
        super().__init__(FastAPIServerConfig, FastAPIServerOperation)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def forward(self, data: FastAPIServerConfig) -> FastAPIServerOperation:
        if not data.routes:
            return FastAPIServerOperation(success=False, reason="No routes provided")

        app = FastAPI()

        @app.get("/healthz")
        def healthz():
            return {"ok": True}

        for r in data.routes:
            method = r.method.upper()
            input_type = r.input
            output_type = r.output
            handler = r.handler

            if not is_dataclass(input_type):
                return FastAPIServerOperation(
                    success=False,
                    reason=f"Route {method} {r.path}: input must be a dataclass type",
                )
            if not is_dataclass(output_type):
                return FastAPIServerOperation(
                    success=False,
                    reason=f"Route {method} {r.path}: output must be a dataclass type",
                )

            async def _endpoint(
                request: Request,
                _method=method,
                _in=input_type,
                _out=output_type,
                _h=handler,
            ):
                try:
                    typed_in = await _parse_request_into(request, _in, _method)

                    res = _h(typed_in)

                    if _is_file_response(res):
                        # Validate contract: must provide path OR content
                        if res.path is None and res.content is None:
                            return JSONResponse(
                                status_code=500,
                                content={
                                    "error": "File response must set either 'path' or 'content'"
                                },
                            )

                        if res.path is not None:
                            return FileResponse(
                                path=str(res.path),
                                media_type=res.media_type,
                                filename=res.filename,
                                headers=dict(res.headers) if res.headers else None,
                                status_code=res.status_code,
                            )
                        # Serve in-memory bytes as a stream
                        headers = dict(res.headers) if res.headers else {}
                        if res.filename and "content-disposition" not in {
                            k.lower() for k in headers
                        }:
                            headers["Content-Disposition"] = (
                                f'attachment; filename="{res.filename}"'
                            )

                            return StreamingResponse(
                                io.BytesIO(res.content or b""),
                                media_type=res.media_type,
                                headers=headers,
                                status_code=res.status_code,
                            )

                    if hasattr(res, "__await__"):
                        res = await res  # type: ignore[misc]

                    if not is_dataclass(res) or not isinstance(res, _out):
                        return JSONResponse(
                            status_code=500,
                            content={
                                "error": "Handler returned wrong type",
                                "expected": _out.__name__,
                                "got": type(res).__name__,
                            },
                        )

                    return asdict(res)
                except Exception as e:
                    return JSONResponse(status_code=400, content={"error": str(e)})

            app.add_api_route(
                path=r.path,
                endpoint=_endpoint,
                methods=[method],
                name=r.name,
                status_code=r.status_code,
            )

        config = uvicorn.Config(
            app=app,
            host=data.host,
            port=data.port,
            log_level=data.log_level,
        )
        self._server = uvicorn.Server(config)

        def _run():
            self._server.run()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

        base_url = f"http://{data.host}:{data.port}"

        deadline = time.time() + data.ready_timeout_s
        last_err: str | None = None
        while time.time() < deadline:
            try:
                resp = requests.get(f"{base_url}/healthz", timeout=0.5)
                if resp.ok:
                    return FastAPIServerOperation(success=True, base_url=base_url)
                last_err = f"healthz status {resp.status_code}"
            except Exception as e:
                last_err = str(e)
            time.sleep(0.15)

        return FastAPIServerOperation(
            success=False,
            reason=f"Timed out waiting for readiness: {last_err}",
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
