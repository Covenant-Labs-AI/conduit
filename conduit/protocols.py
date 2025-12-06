from typing import (
    Protocol,
    runtime_checkable,
)


@runtime_checkable
class SupportsSqlCommand(Protocol):
    sql_command: str


@runtime_checkable
class SupportsFileContent(Protocol):
    file_content: str
