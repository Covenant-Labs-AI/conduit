import uuid
from typing import Optional, Iterable, Union
from sqlmodel import SQLModel, Field, Session, create_engine, select
from sqlalchemy.orm import selectinload  # or joinedload
from conduit.conduit_types import (
    GPUS,
    ComputeProvider,
    DeploymentStatus,
    DeploymentType,
    Runtime,
)
from conduit.state.db import Deployment, get_session


def create_deployment(
    deployment_key: str,
    image: str,
    runtime: Runtime,
    gpu: GPUS,
    deployment_type: DeploymentType,
    provider: ComputeProvider,
    gpu_count: int | None = None,
    ports: str | None = None,
    replicas: int | None = 1,
) -> Deployment:

    with get_session() as s:
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
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return obj


def get_deployment(deployment_key: str) -> Optional[Deployment]:
    with get_session() as s:
        statement = (
            select(Deployment)
            .where(Deployment.deployment_key == deployment_key)
            .options(selectinload(Deployment.nodes))  # or joinedload
        )
        deployment = s.exec(statement).first()
        # force loading while session is open (optional but explicit)
        if deployment is not None:
            _ = deployment.nodes  # triggers the load inside the session
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
    """
    Usage: update_deployment(id, status=DeploymentStatus.DEPLOYED, public_endpoint="http://...")
    Only provided fields are updated.
    """
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


# --- Example quick test ---
if __name__ == "__main__":
    # CREATE
    d = create_deployment(
        image="myregistry/model:v1",
        image_type=DeploymentType.LLM,
        provider=ComputeProvider.RUNPOD,
        status=DeploymentStatus.DEPLOYING,
        public_endpoint="http://123.45.67.89",
        ports="80,443",
        tensor_parallel_size=2,
    )
    print("Created:", d.id)

    # READ
    print("Get:", get_deployment(d.id))
    print("List (all):", list_deployments())

    # UPDATE
    updated = update_deployment(d.id, status=DeploymentStatus.DEPLOYED)
    print("Updated:", updated.status)

    # DELETE
    print("Deleted:", delete_deployment(d.id))
