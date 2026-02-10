import os
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlmodel import (
    SQLModel,
    Field,
    create_engine,
    Session,
    Relationship,
    Column,
    JSON,
)
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url, URL
from sqlalchemy import event
from sqlalchemy.pool import StaticPool

from conduit.conduit_types import (
    ComputeProvider,
    DeploymentStatus,
    DeploymentType,
    NodeStatus,
    Runtime,
)


class Deployment(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    deployment_key: str
    image: str
    gpu: str
    runtime: Runtime
    deployment_type: DeploymentType
    provider: ComputeProvider
    status: DeploymentStatus = Field(default=DeploymentStatus.DEPLOYING)
    ports: Optional[str] = None
    gpu_count: Optional[int] = 1
    replicas: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    nodes: List["Node"] = Relationship(
        back_populates="deployment",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Node(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    external_id: str = Field(unique=True)
    ip_address: Optional[str] = Field(default=None)
    deployment_id: uuid.UUID = Field(foreign_key="deployment.id")
    deployment: Deployment = Relationship(back_populates="nodes")
    status: NodeStatus = Field(default=NodeStatus.PROVISIONING)

    port_map: Optional[Dict[str, int]] = Field(default=None, sa_column=Column(JSON))

    def resolve_port(self, internal_port: int) -> Optional[int]:
        if not self.port_map:
            return internal_port

        return self.port_map.get(str(internal_port), internal_port)


@dataclass(frozen=True)
class DBConfig:
    db_uri: str
    dialect: str

    echo: bool = False
    pool_pre_ping: bool = True
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 1800
    connect_timeout: int = 5
    application_name: str = "conduit"

    sqlite_check_same_thread: bool = False
    sqlite_wal: bool = True
    sqlite_foreign_keys: bool = True
    sqlite_synchronous: str = "NORMAL"

    def summary(self) -> Dict[str, Any]:
        d = asdict(self)
        try:
            url = make_url(self.db_uri)
            d["db_uri_redacted"] = str(url._replace(password="***"))
        except Exception:
            d["db_uri_redacted"] = "<unparseable>"
        d.pop("db_uri", None)
        return d


def _load_config_from_env() -> DBConfig:
    db_uri = os.getenv("CONDUIT_DB_URI", "sqlite:///conduit.db")
    echo = os.getenv("CONDUIT_DB_ECHO", "0") == "1"
    app_name = os.getenv("CONDUIT_APP_NAME", "conduit")

    url = make_url(db_uri)
    dialect = url.get_backend_name()

    if dialect == "sqlite":
        return DBConfig(
            db_uri=db_uri,
            dialect=dialect,
            echo=echo,
            application_name=app_name,
            sqlite_wal=os.getenv("CONDUIT_SQLITE_WAL", "1") == "1",
            sqlite_foreign_keys=os.getenv("CONDUIT_SQLITE_FK", "1") == "1",
            sqlite_synchronous=os.getenv("CONDUIT_SQLITE_SYNC", "NORMAL"),
        )

    return DBConfig(
        db_uri=db_uri,
        dialect=dialect,
        echo=echo,
        application_name=app_name,
        pool_pre_ping=os.getenv("CONDUIT_POOL_PRE_PING", "1") == "1",
        pool_size=int(os.getenv("CONDUIT_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("CONDUIT_MAX_OVERFLOW", "10")),
        pool_recycle=int(os.getenv("CONDUIT_POOL_RECYCLE", "1800")),
        connect_timeout=int(os.getenv("CONDUIT_CONNECT_TIMEOUT", "5")),
    )


def _install_sqlite_pragmas(engine: Engine, cfg: DBConfig) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        if cfg.sqlite_foreign_keys:
            cur.execute("PRAGMA foreign_keys=ON;")
        if cfg.sqlite_wal:
            cur.execute("PRAGMA journal_mode=WAL;")
        if cfg.sqlite_synchronous:
            cur.execute(f"PRAGMA synchronous={cfg.sqlite_synchronous};")
        cur.close()


def create_configured_engine(cfg: DBConfig) -> Engine:
    url = make_url(cfg.db_uri)

    if cfg.dialect == "sqlite":
        is_memory = str(url.database or "") in (
            "",
            ":memory:",
        ) and cfg.db_uri.startswith("sqlite:")
        engine = create_engine(
            cfg.db_uri,
            echo=cfg.echo,
            connect_args={
                "check_same_thread": cfg.sqlite_check_same_thread is True
                and True
                or False
            },
            poolclass=StaticPool if is_memory else None,
        )
        _install_sqlite_pragmas(engine, cfg)
        return engine

    connect_args = {
        "connect_timeout": cfg.connect_timeout,
        "application_name": cfg.application_name,
    }

    return create_engine(
        cfg.db_uri,
        echo=cfg.echo,
        pool_pre_ping=cfg.pool_pre_ping,
        pool_size=cfg.pool_size,
        max_overflow=cfg.max_overflow,
        pool_recycle=cfg.pool_recycle,
        connect_args=connect_args,
    )


_CFG = _load_config_from_env()
ENGINE: Engine = create_configured_engine(_CFG)


def get_session() -> Session:
    return Session(ENGINE)


# TODO future releases will have migrations based on conduit version
SQLModel.metadata.create_all(ENGINE)
