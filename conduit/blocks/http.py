import requests
from dataclasses import dataclass, is_dataclass, asdict
from typing import (
    Type,
    Any,
    cast,
)

from dataclasses import dataclass, is_dataclass, asdict, fields
from typing import Type, Any


from .block_types import Block, I, O, NoOp


@dataclass
class HttpOperation:
    success: bool
    status_code: int | None = None
    data: str | None = None
    reason: str | None = None


class HttpGetBlock(Block[NoOp, HttpOperation]):
    def __init__(
        self,
        endpoint: str,
        headers: dict | None = None,
    ):
        super().__init__(NoOp, HttpOperation)
        self.endpoint = endpoint
        self.headers = headers or {}

    def forward(self, data: NoOp) -> HttpOperation:
        try:
            resp = requests.get(self.endpoint, headers=self.headers)
            if resp.ok:
                return HttpOperation(
                    success=True,
                    status_code=resp.status_code,
                    data=resp.text,
                )
            else:
                return HttpOperation(
                    success=False,
                    status_code=resp.status_code,
                    reason=resp.text,
                )
        except Exception as e:
            return HttpOperation(success=False, reason=str(e))

    def __call__(self, data: NoOp | None = None) -> HttpOperation:
        if data is None:
            data = NoOp()
        return super().__call__(data)


class HttpPostBlock(Block[I, HttpOperation]):
    def __init__(self, input: Type[I], endpoint: str, headers: dict | None = None):
        super().__init__(input, HttpOperation)

        self.endpoint = endpoint
        self.input = input
        self.headers = headers or {}

    def forward(self, data: I) -> HttpOperation:
        try:
            if is_dataclass(data):
                payload = asdict(cast(Any, data))
            else:
                return HttpOperation(
                    success=True,
                    status_code=500,
                    data="Data input must be a Dataclass",
                )
            resp = requests.post(self.endpoint, json=payload, headers=self.headers)
            if resp.ok:
                return HttpOperation(
                    success=True,
                    status_code=resp.status_code,
                    data=resp.text,
                )
            else:
                return HttpOperation(
                    success=False,
                    status_code=resp.status_code,
                    reason=resp.text,
                )
        except Exception as e:
            return HttpOperation(success=False, reason=str(e))
