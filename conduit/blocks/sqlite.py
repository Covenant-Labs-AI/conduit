import sqlite3
from pathlib import Path

from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .block_types import Block


DbOp = Literal["insert", "insert_many", "update", "delete", "select"]


@dataclass(frozen=True)
class DbAction:
    """
    A compact, structured representation of a database operation.

    - No raw SQL values are ever inlined into the SQL string.
    - WHERE is equality-only to keep it safe and simple.
    - For update/delete, WHERE is required by default (configurable).
    """

    op: DbOp
    table: str

    # For insert/update: column -> value
    values: Mapping[str, Any] | None = None

    # For insert_many: explicit columns + rows
    columns: Sequence[str] | None = None
    rows: Sequence[Sequence[Any]] | None = None

    # Equality-only WHERE: column -> value
    where: Mapping[str, Any] | None = None

    # For select: columns to return (defaults to ["*"] if None)
    select_columns: Sequence[str] | None = None

    # For select: optional LIMIT
    limit: int | None = None


@runtime_checkable
class SupportsDbAction(Protocol):
    db_action: DbAction


@dataclass
class SqlOperation:
    success: bool
    reason: str | None = None

    # For SELECT only
    rows: list[tuple[Any, ...]] | None = None
    columns: list[str] | None = None

    # Basic info
    rowcount: int | None = None
    lastrowid: int | None = None


class Sqlite3Block[Input: SupportsDbAction](Block[Input, SqlOperation]):
    """
    Security posture:
      - No arbitrary SQL accepted.
      - Table/column names validated against the loaded schema.
      - Equality-only WHERE clauses.
      - Optional allowlist of tables.
      - Optional requirement for WHERE on update/delete.

    Notes:
      - This block assumes your schema is applied at init time (schema.sql).
      - It introspects SQLite schema after applying schema.sql to validate
        tables/columns.
    """

    def __init__(
        self,
        input: type[Input],
        database_url: str = ":memory:",
        schema_file: str = "schema.sql",
        *,
        allowed_tables: set[str] | None = None,
        require_where_for_update_delete: bool = True,
        max_insert_many_rows: int = 5_000,
        max_select_limit: int = 10_000,
    ):
        super().__init__(input, SqlOperation)
        self.conn = sqlite3.connect(database_url)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        self.allowed_tables = allowed_tables
        self.require_where_for_update_delete = require_where_for_update_delete
        self.max_insert_many_rows = max_insert_many_rows
        self.max_select_limit = max_select_limit

        # Load and apply schema
        try:
            with open(schema_file, "r", encoding="utf-8") as f:
                schema = f.read()
            self.cursor.executescript(schema)
            self.conn.commit()
        except FileNotFoundError:
            raise RuntimeError(f"Schema file not found: {schema_file}")
        except sqlite3.DatabaseError as e:
            raise RuntimeError(f"Failed to apply schema from {schema_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading schema: {e}")

        # Introspect schema for validation (table -> set(columns))
        self._schema_map = self._introspect_schema()

        # If allowlist is provided, validate it
        if self.allowed_tables is not None:
            unknown = self.allowed_tables - set(self._schema_map.keys())
            if unknown:
                raise RuntimeError(
                    f"allowed_tables contains unknown tables: {sorted(unknown)}"
                )

    # ------------------------- Schema / Validation -------------------------

    def _introspect_schema(self) -> dict[str, set[str]]:
        """
        Build a map of {table_name: {column_name, ...}} from sqlite_master + PRAGMA table_info.
        """
        tables = []
        self.cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            """
        )
        for row in self.cursor.fetchall():
            tables.append(row[0])

        schema_map: dict[str, set[str]] = {}
        for t in tables:
            self.cursor.execute(f"PRAGMA table_info({t})")
            cols = {r[1] for r in self.cursor.fetchall()}  # r[1] is column name
            schema_map[t] = cols

        return schema_map

    def _validate_table(self, table: str) -> None:
        if table not in self._schema_map:
            raise ValueError(f"Table not found in schema: {table}")
        if self.allowed_tables is not None and table not in self.allowed_tables:
            raise ValueError(f"Table not allowed: {table}")

    def _validate_columns(self, table: str, cols: Sequence[str]) -> None:
        allowed_cols = self._schema_map[table]
        bad = [c for c in cols if c not in allowed_cols and c != "*"]
        if bad:
            raise ValueError(f"Column(s) not allowed for table '{table}': {bad}")

    def _require_mapping(
        self, m: Mapping[str, Any] | None, label: str
    ) -> Mapping[str, Any]:
        if m is None or len(m) == 0:
            raise ValueError(f"Missing or empty '{label}'")
        return m

    # ------------------------- SQL Builders -------------------------

    def _build_insert(self, a: DbAction) -> tuple[str, list[Any]]:
        values = self._require_mapping(a.values, "values")
        cols = list(values.keys())
        self._validate_columns(a.table, cols)
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO {a.table} ({', '.join(cols)}) VALUES ({placeholders})"
        params = [values[c] for c in cols]
        return sql, params

    def _build_insert_many(self, a: DbAction) -> tuple[str, list[Sequence[Any]]]:
        if not a.columns or len(a.columns) == 0:
            raise ValueError("insert_many requires 'columns'")
        if not a.rows or len(a.rows) == 0:
            raise ValueError("insert_many requires non-empty 'rows'")
        if len(a.rows) > self.max_insert_many_rows:
            raise ValueError(
                f"insert_many rows exceeds max_insert_many_rows={self.max_insert_many_rows}"
            )

        cols = list(a.columns)
        self._validate_columns(a.table, cols)

        width = len(cols)
        for i, r in enumerate(a.rows):
            if len(r) != width:
                raise ValueError(
                    f"Row {i} length {len(r)} does not match columns length {width}"
                )

        placeholders = ", ".join(["?"] * width)
        sql = f"INSERT INTO {a.table} ({', '.join(cols)}) VALUES ({placeholders})"
        return sql, list(a.rows)

    def _build_where(
        self, table: str, where: Mapping[str, Any] | None
    ) -> tuple[str, list[Any]]:
        if where is None or len(where) == 0:
            return "", []
        cols = list(where.keys())
        self._validate_columns(table, cols)
        clause = " AND ".join([f"{c} = ?" for c in cols])
        params = [where[c] for c in cols]
        return f" WHERE {clause}", params

    def _build_update(self, a: DbAction) -> tuple[str, list[Any]]:
        values = self._require_mapping(a.values, "values")
        if self.require_where_for_update_delete and (
            a.where is None or len(a.where) == 0
        ):
            raise ValueError("UPDATE requires 'where' (safety requirement)")

        set_cols = list(values.keys())
        self._validate_columns(a.table, set_cols)
        set_clause = ", ".join([f"{c} = ?" for c in set_cols])
        set_params = [values[c] for c in set_cols]

        where_sql, where_params = self._build_where(a.table, a.where)
        sql = f"UPDATE {a.table} SET {set_clause}{where_sql}"
        return sql, set_params + where_params

    def _build_delete(self, a: DbAction) -> tuple[str, list[Any]]:
        if self.require_where_for_update_delete and (
            a.where is None or len(a.where) == 0
        ):
            raise ValueError("DELETE requires 'where' (safety requirement)")
        where_sql, where_params = self._build_where(a.table, a.where)
        sql = f"DELETE FROM {a.table}{where_sql}"
        return sql, where_params

    def _build_select(self, a: DbAction) -> tuple[str, list[Any]]:
        cols = list(a.select_columns) if a.select_columns else ["*"]
        self._validate_columns(a.table, cols)

        where_sql, where_params = self._build_where(a.table, a.where)

        limit = a.limit
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be > 0")
            if limit > self.max_select_limit:
                raise ValueError(
                    f"limit exceeds max_select_limit={self.max_select_limit}"
                )

        limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT {', '.join(cols)} FROM {a.table}{where_sql}{limit_sql}"
        return sql, where_params

    def _compile_action(self, a: DbAction) -> tuple[str, Any, bool]:
        """
        Returns: (sql, params_or_rows, is_many)
        """
        self._validate_table(a.table)

        if a.op == "insert":
            sql, params = self._build_insert(a)
            return sql, params, False

        if a.op == "insert_many":
            sql, rows = self._build_insert_many(a)
            return sql, rows, True

        if a.op == "update":
            sql, params = self._build_update(a)
            return sql, params, False

        if a.op == "delete":
            sql, params = self._build_delete(a)
            return sql, params, False

        if a.op == "select":
            sql, params = self._build_select(a)
            return sql, params, False

        raise ValueError(f"Unsupported operation: {a.op}")

    # ------------------------- Block Interface -------------------------

    def forward(self, data: Input) -> SqlOperation:
        if not hasattr(data, "db_action"):
            return SqlOperation(
                success=False,
                reason="SQL block input dataclass must have a 'db_action' field",
            )

        a = data.db_action

        try:
            sql, params_or_rows, is_many = self._compile_action(a)

            if is_many:
                self.cursor.executemany(sql, params_or_rows)
            else:
                self.cursor.execute(sql, params_or_rows)

            # SELECT returns rows; others just commit
            if a.op == "select":
                fetched = self.cursor.fetchall()
                cols = list(fetched[0].keys()) if fetched else []
                rows = [tuple(r) for r in fetched]
                return SqlOperation(
                    success=True,
                    rows=rows,
                    columns=cols,
                    rowcount=self.cursor.rowcount,
                    lastrowid=None,
                )

            self.conn.commit()
            return SqlOperation(
                success=True,
                rowcount=self.cursor.rowcount,
                lastrowid=self.cursor.lastrowid,
            )

        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            return SqlOperation(success=False, reason=str(e))

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
