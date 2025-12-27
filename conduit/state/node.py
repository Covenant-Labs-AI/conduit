import uuid
from typing import Dict
from sqlmodel import SQLModel, Field, Session, create_engine, select
from conduit.conduit_types import DeploymentStatus, NodeStatus
from conduit.state.db import Node, get_session


def create_node(
    *,
    session: Session,
    external_id: str | None,
    ip_address: str | None,
    port_map: dict | None,
    deployment_id: uuid.UUID,
) -> Node:
    node = Node(
        external_id=external_id,
        ip_address=ip_address,
        port_map=port_map,
        deployment_id=deployment_id,
        status=NodeStatus.PROVISIONING,
    )
    session.add(node)
    session.flush()
    return node


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
            select(Node)
            .where(Node.deployment_id == deployment_id)
            .where(Node.status == NodeStatus.DEPLOYED)
        )
        return list(s.exec(statement))


def get_nodes_by_deployment(deployment_id: uuid.UUID) -> list[Node]:
    with get_session() as s:
        statement = select(Node).where(Node.deployment_id == deployment_id)
        return list(s.exec(statement))
