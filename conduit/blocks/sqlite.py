import requests
import subprocess
import sqlite3
from dataclasses import dataclass, is_dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TypeVar,
    Generic,
    TypeGuard,
    Type,
    Any,
    Protocol,
    Mapping,
    Dict,
    List,
    Literal,
    runtime_checkable,
    cast,
)


import threading
import time
from dataclasses import dataclass, is_dataclass, asdict, fields
from typing import List, Literal, Type, TypeVar, Callable, Any, Optional, get_type_hints

import io
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse


from .block_types import Block, I, O


@runtime_checkable
class SupportsSqlCommand(Protocol):
    sql_command: str


@dataclass
class SqlOperation:
    success: bool
    reason: str | None


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
