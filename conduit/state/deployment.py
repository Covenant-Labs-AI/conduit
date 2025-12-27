import uuid
from typing import Optional, Iterable, Union
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy.orm import selectinload  # or joinedload
from conduit.conduit_types import (
    ComputeProvider,
    DeploymentStatus,
    DeploymentType,
    Runtime,
)
from conduit.state.db import Deployment, get_session


def create_deployment(
    *,
    session: Session,
    deployment_key: str,
    image: str,
    runtime: Runtime,
    gpu: str,
    deployment_type: DeploymentType,
    provider: ComputeProvider,
    gpu_count: int | None = None,
    ports: str | None = None,
    replicas: int | None = 1,
) -> Deployment:
    obj = Deployment(
        deployment_key=deployment_key,
        image=image,
        runtime=runtime,
        gpu_count=gpu_count,
        gpu=gpu,
        deployment_type=deployment_type,
        provider=provider,
        status=DeploymentStatus.DEPLOYING,
        replicas=replicas,
        ports=ports,
    )
    session.add(obj)
    session.flush()
    return obj


def get_deployment(deployment_key: str) -> Optional[Deployment]:
    with get_session() as s:
        statement = (
            select(Deployment)
            .where(Deployment.deployment_key == deployment_key)
            .options(selectinload(Deployment.nodes))
        )
        deployment = s.exec(statement).first()
        if deployment is not None:
            _ = deployment.nodes
        return deployment


def find_by_image_name(image: str) -> Optional[Deployment]:
    with get_session() as s:
        statement = select(Deployment).where(Deployment.image == image)
        result = s.exec(statement).first()
        return result


def list_deployments(limit: int = 100, **filters) -> list[Deployment]:
    with get_session() as s:
        stmt = select(Deployment)

        for field, value in filters.items():
            if hasattr(Deployment, field) and value is not None:
                stmt = stmt.where(getattr(Deployment, field) == value)

        stmt = stmt.limit(limit)
        return s.exec(stmt).all()


def update_deployment_status(
    deployment_id: uuid.UUID,
    status: DeploymentStatus,
) -> Deployment | None:
    with get_session() as s:
        deployment = s.get(Deployment, deployment_id)
        if not deployment:
            return None

        deployment.status = status

        s.add(deployment)
        s.commit()
        s.refresh(deployment)
        return deployment


def update_deployment(
    deployment_id: Union[str, uuid.UUID],
    **fields,
) -> Optional[Deployment]:
    did = uuid.UUID(str(deployment_id))
    with get_session() as s:
        obj = s.get(Deployment, did)
        if not obj:
            return None
        # whitelist to avoid accidental/invalid writes
        allowed: set[str] = {
            "image",
            "image_type",
            "provider",
            "status",
            "public_endpoint",
            "ports",
            "tensor_parallel_size",
        }
        for k, v in fields.items():
            if k in allowed:
                setattr(obj, k, v)
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return obj


def delete_deployment_by_key(deployment_key: str) -> bool:
    with get_session() as s:
        stmt = select(Deployment).where(Deployment.deployment_key == deployment_key)
        obj = s.exec(stmt).first()

        if not obj:
            return False

        s.delete(obj)
        s.commit()
        return True


def delete_deployment(deployment_id: Union[str, uuid.UUID]) -> bool:
    did = uuid.UUID(str(deployment_id))
    with get_session() as s:
        obj = s.get(Deployment, did)
        if not obj:
            return False
        s.delete(obj)
        s.commit()
        return True
