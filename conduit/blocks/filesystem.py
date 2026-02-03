from dataclasses import dataclass
from pathlib import Path
from typing import (
    Type,
    Protocol,
    runtime_checkable,
)

from dataclasses import dataclass
from typing import Type


from .block_types import Block, I, O, NoOp


@runtime_checkable
class SupportsFileContent(Protocol):
    file_content: str


@dataclass
class FileSystemOperation:
    success: bool
    error_code: int | None = None
    data: bytes | None = None
    reason: str | None = None
    path: Path | None = None


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
