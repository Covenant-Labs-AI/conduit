import uuid
from typing import Dict
from sqlmodel import SQLModel, Field, Session, create_engine, select
from conduit.conduit_types import DeploymentStatus, NodeStatus
from conduit.state.db import Node, get_session


def create_node(
    external_id: str,
    ip_address: str | None,
    port_map: Dict[str, int] | None,
    deployment_id: uuid.UUID,
) -> Node:
    with get_session() as s:
        obj = Node(
            port_map=port_map,
            external_id=external_id,
            ip_address=ip_address,
            deployment_id=deployment_id,
        )
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return obj


def update_node_info(
    node_id: uuid.UUID,
    status: NodeStatus,
    ip_address: str | None,
    port_map: Dict[str, int] | None,
) -> Node | None:
    with get_session() as s:
        node = s.get(Node, node_id)
        if not node:
            return None

        node.status = status
        node.ip_address = ip_address
        node.port_map = port_map

        s.add(node)
        s.commit()
        s.refresh(node)
        return node


def update_node_status(node_id: uuid.UUID, status: NodeStatus) -> Node | None:
    with get_session() as s:
        node = s.get(Node, node_id)
        if not node:
            return None

        node.status = status
        s.add(node)
        s.commit()
        s.refresh(node)

        return node


def get_node_by_id(node_id: uuid.UUID) -> Node | None:
    with get_session() as s:
        return s.get(Node, node_id)


def get_node_by_external_id(external_id: str) -> Node | None:
    with get_session() as s:
        statement = select(Node).where(Node.external_id == external_id)
        return s.exec(statement).first()


def get_ready_nodes_by_deployment(deployment_id: uuid.UUID) -> list[Node]:
    with get_session() as s:
        statement = (
            select(Node).where(Node.deployment_id == deployment_id)
            # .where(Node.status == DeploymentStatus.DEPLOYED)
        )
        return list(s.exec(statement))


def get_nodes_by_deployment(deployment_id: uuid.UUID) -> list[Node]:
    with get_session() as s:
        statement = select(Node).where(Node.deployment_id == deployment_id)
        return list(s.exec(statement))
