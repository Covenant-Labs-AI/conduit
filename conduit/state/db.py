import os
import uuid
from typing import Optional, List, Dict
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

from conduit.conduit_types import (
    GPUS,
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
    gpu: GPUS
    runtime: Runtime
    deployment_type: DeploymentType
    provider: ComputeProvider
    status: DeploymentStatus = Field(default=DeploymentStatus.DEPLOYING)
    ports: Optional[str] = None  # csv
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


DATABASE_URL = os.getenv("CONDUIT_DB_URI", "sqlite:///conduit.db")

engine = create_engine(DATABASE_URL, echo=False)

engine = create_engine("sqlite:///conduit.db", echo=False)

SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
