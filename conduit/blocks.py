import requests
import sqlite3
from dataclasses import dataclass, is_dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TypeVar,
    Generic,
    Type,
    Any,
    Protocol,
    Dict,
    List,
    Literal,
    runtime_checkable,
    cast,
)

I = TypeVar("I")
O = TypeVar("O")


@runtime_checkable
class SupportsSqlCommand(Protocol):
    sql_command: str


@runtime_checkable
class SupportsFileContent(Protocol):
    file_content: str


@dataclass
class NoOp:
    """Placeholder dataclass for Conduit blocks that take no input or output"""

    pass


@dataclass
class HttpOperation:
    success: bool
    status_code: int | None = None
    data: str | None = None
    reason: str | None = None


@dataclass
class SqlOperation:
    success: bool
    reason: str | None


@dataclass
class FileSystemOperation:
    success: bool
    error_code: int | None = None
    data: bytes | None = None
    reason: str | None = None
    path: Path | None = None


@dataclass
class FinetuneOperation:
    success: bool
    message: str


class Block(Generic[I, O], ABC):
    def __init__(self, input: Type[I], output: Type[O]):
        self.input = input
        self.output = output

    def __call__(self, data: I) -> O:
        if not is_dataclass(data) or not isinstance(data, self.input):
            raise TypeError(
                f"Expected dataclass instance of {self.input.__name__}, "
                f"got {type(data).__name__}"
            )
        out = self.forward(data)
        if not is_dataclass(out) or not isinstance(out, self.output):
            raise TypeError(
                f"forward() must return {self.output.__name__} dataclass instance, "
                f"got {type(out).__name__}"
            )
        return out

    @abstractmethod
    def forward(self, data: I) -> O:
        """Transform input -> output."""

        raise NotImplementedError


class Sqlite3Block[Input: SupportsSqlCommand](Block[Input, SqlOperation]):
    def __init__(
        self,
        input: type[Input],
        database_url: str = ":memory:",
        schema_file: str = "schema.sql",
    ):
        super().__init__(input, SqlOperation)
        self.conn = sqlite3.connect(database_url)
        self.cursor = self.conn.cursor()

        try:
            with open(schema_file, "r") as f:
                schema = f.read()
            self.cursor.executescript(schema)
            self.conn.commit()
        except FileNotFoundError:
            raise RuntimeError(f"Schema file not found: {schema_file}")
        except sqlite3.DatabaseError as e:
            raise RuntimeError(f"Failed to apply schema from {schema_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading schema: {e}")

    def forward(self, data: Input) -> SqlOperation:
        if not hasattr(data, "sql_command"):
            return SqlOperation(
                success=False,
                reason="SQL block input dataclass must have a 'sql_command' field",
            )

        try:
            self.cursor.execute(data.sql_command)
            self.conn.commit()
            return SqlOperation(success=True, reason=None)
        except Exception as e:
            return SqlOperation(success=False, reason=str(e))

    def __del__(self):
        self.conn.close()


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


class FileSystemReadBlock(Block[NoOp, FileSystemOperation]):
    def __init__(self, path: Path, mode="r"):
        self.path = path
        self.mode = mode
        super().__init__(NoOp, FileSystemOperation)

    def forward(self, data: NoOp) -> FileSystemOperation:
        try:
            with self.path.open(mode=self.mode) as f:
                content = f.read()
            return FileSystemOperation(
                success=True,
                error_code=None,
                data=content,
                reason=None,
                path=self.path,
            )
        except Exception as e:
            return FileSystemOperation(
                success=False,
                error_code=getattr(e, "errno", None),
                data=None,
                reason=str(e),
                path=self.path,
            )

    def __call__(self, data: NoOp | None = None) -> FileSystemOperation:
        if data is None:
            data = NoOp()
        return super().__call__(data)


class FileSystemWriteBlock[Input: SupportsFileContent](
    Block[Input, FileSystemOperation]
):
    def __init__(self, input: Type[Input], path: Path, mode: str = "w"):
        self.path = path
        self.mode = mode
        super().__init__(input, FileSystemOperation)

    def forward(self, data: Input) -> FileSystemOperation:
        if not hasattr(data, "file_content"):
            return FileSystemOperation(
                success=False,
                reason="Input dataclass must have a 'file_content' field",
                path=self.path,
            )
        try:
            with self.path.open(mode=self.mode) as f:
                f.write(data.file_content)  # assume correct type for mode
            return FileSystemOperation(success=True, path=self.path)
        except Exception as e:
            return FileSystemOperation(
                success=False,
                error_code=getattr(e, "errno", None),
                reason=str(e),
                path=self.path,
            )
