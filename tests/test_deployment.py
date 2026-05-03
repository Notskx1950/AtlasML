"""Tests for the model deployment API."""

from __future__ import annotations

import hashlib

import pytest
from httpx import AsyncClient
# --- helper functions for tests ---
def create_fake_artifact(tmp_path, filename: str, content: bytes | None = None) -> str:
    path = tmp_path / filename
    path.write_bytes(content or b"fake model artifact")
    return str(path)

# --- Tests ---
@pytest.mark.asyncio
async def test_activate_v1_creates_deployment_event(client: AsyncClient, tmp_path) -> None:
    artifact_uri = create_fake_artifact(tmp_path, "event_model.joblib")

    await client.post(
        "/models/register",
        json={
            "name": "event-model",
            "version": "v1",
            "artifact_uri": artifact_uri,
        },
    )

    response = await client.post(
        "/models/event-model/activate",
        json={"version": "v1", "reason": "initial activation"},
    )

    assert response.status_code == 200
    assert response.json()["is_active"] is True

    response = await client.get("/models/event-model/deployments")
    assert response.status_code == 200

    events = response.json()
    assert len(events) == 1

    event = events[0]
    assert event["model_name"] == "event-model"
    assert event["from_version"] is None
    assert event["to_version"] == "v1"
    assert event["action"] == "activate"
    assert event["reason"] == "initial activation"
    assert event["created_at"] is not None

@pytest.mark.asyncio
async def test_activate_v2_records_previous_active_version(client: AsyncClient, tmp_path) -> None:
    for v in ["v1", "v2"]:
        artifact_uri = create_fake_artifact(tmp_path, f"event_{v}.joblib")
        await client.post(
            "/models/register",
            json={"name": "event-rollback", "version": v, "artifact_uri": artifact_uri},
        )

    response = await client.post("/models/event-rollback/activate", json={"version": "v1", "reason": "initial activation"})
    assert response.status_code == 200

    response = await client.post("/models/event-rollback/activate", json={"version": "v2", "reason": "upgrade"})
    assert response.status_code == 200

    response = await client.get("/models/event-rollback/deployments")
    assert response.status_code == 200

    events = response.json()
    assert len(events) == 2

    latest = events[0]
    assert latest["from_version"] == "v1"
    assert latest["to_version"] == "v2"
    assert latest["action"] == "activate"
    assert latest["reason"] == "upgrade"
    
@pytest.mark.asyncio
async def test_rollback_records_deployment_event(client: AsyncClient, tmp_path) -> None:
    for v in ["v1", "v2"]:
        artifact_uri = create_fake_artifact(tmp_path, f"rb_event_{v}.joblib")
        await client.post(
            "/models/register",
            json={"name": "rb-event", "version": v, "artifact_uri": artifact_uri},
        )

    response = await client.post("/models/rb-event/activate", json={"version": "v1", "reason": "initial activation"})
    assert response.status_code == 200

    response = await client.post("/models/rb-event/activate", json={"version": "v2", "reason": "upgrade"})
    assert response.status_code == 200

    response = await client.post("/models/rb-event/activate", json={"version": "v1", "action": "rollback", "reason": "issue with v2"})
    assert response.status_code == 200

    response = await client.get("/models/rb-event/deployments")
    assert response.status_code == 200

    events = response.json()
    assert len(events) == 3

    latest = events[0]
    assert latest["from_version"] == "v2"
    assert latest["to_version"] == "v1"
    assert latest["action"] == "rollback"
    assert latest["reason"] == "issue with v2"

@pytest.mark.asyncio
async def test_activate_missing_version_does_not_deactivate_current_active(client: AsyncClient, tmp_path) -> None:
    artifact_uri = create_fake_artifact(tmp_path, "active_model.joblib")

    await client.post(
        "/models/register",
        json={
            "name": "active-model",
            "version": "v1",
            "artifact_uri": artifact_uri,
        },
    )

    response = await client.post("/models/active-model/activate", json={"version": "v1"})
    assert response.status_code == 200
    assert response.json()["is_active"] is True

    response = await client.post("/models/active-model/activate", json={"version": "nonexistent"})
    assert response.status_code == 404

    response = await client.get("/models/active-model/deployments")
    assert response.status_code == 200
    assert len(response.json()) == 1

@pytest.mark.asyncio
async def test_idempotent_activation_does_not_create_duplicate_events(client: AsyncClient, tmp_path) -> None:
    artifact_uri = create_fake_artifact(tmp_path, "idempotent_model.joblib")

    await client.post(
        "/models/register",
        json={
            "name": "idempotent-model",
            "version": "v1",
            "artifact_uri": artifact_uri,
        },
    )

    response = await client.post("/models/idempotent-model/activate", json={"version": "v1", "reason": "initial activation"})
    assert response.status_code == 200

    response = await client.post("/models/idempotent-model/activate", json={"version": "v1", "reason": "initial activation"})
    assert response.status_code == 200

    response = await client.get("/models/idempotent-model/deployments")
    assert response.status_code == 200
    events = response.json()
    assert len(events) == 1
    assert events[0]["action"] == "activate"