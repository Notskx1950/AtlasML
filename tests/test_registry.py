"""Tests for the model registry API."""

from __future__ import annotations

import hashlib

import pytest
from httpx import AsyncClient

# --- helper functions for tests ---
def create_fake_artifact(tmp_path, filename: str, content: bytes | None = None) -> str:
    path = tmp_path / filename
    path.write_bytes(content or b"fake model artifact")
    return str(path)


# --- tests ---
@pytest.mark.asyncio
async def test_register_model(client: AsyncClient, tmp_path) -> None:
    """Register a model version and verify the response."""
    artifact_uri = create_fake_artifact(tmp_path, "test.joblib")
    resp = await client.post(
        "/models/register",
        json={
            "name": "test-model",
            "version": "v1.0",
            "artifact_uri": artifact_uri,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test-model"
    assert data["version"] == "v1.0"
    assert data["is_active"] is False
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_activate_model(client: AsyncClient, tmp_path) -> None:
    """Activate a version and verify only it is active."""
    # Register two versions
    artifact_uri = create_fake_artifact(tmp_path, "v1.joblib")
    await client.post(
        "/models/register",
        json={"name": "act-model", "version": "v1", "artifact_uri": artifact_uri},
    )
    artifact_uri = create_fake_artifact(tmp_path, "v2.joblib")
    await client.post(
        "/models/register",
        json={"name": "act-model", "version": "v2", "artifact_uri": artifact_uri},
    )

    # Activate v1
    resp = await client.post(
        "/models/act-model/activate",
        json={"version": "v1"},
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True

    # Verify v1 is the only active version
    resp = await client.get("/models/act-model/active")
    assert resp.status_code == 200
    assert resp.json()["version"] == "v1"


@pytest.mark.asyncio
async def test_rollback_activation(client: AsyncClient, tmp_path) -> None:
    """Activate v1, then v2, then rollback to v1."""
    for v in ["v1", "v2"]:
        artifact_uri = create_fake_artifact(tmp_path, f"{v}.joblib")
        await client.post(
            "/models/register",
            json={"name": "rb-model", "version": v, "artifact_uri": artifact_uri},
        )

    await client.post("/models/rb-model/activate", json={"version": "v1"})
    await client.post("/models/rb-model/activate", json={"version": "v2"})

    # Verify v2 is active
    resp = await client.get("/models/rb-model/active")
    assert resp.json()["version"] == "v2"

    # Rollback to v1
    await client.post("/models/rb-model/activate", json={"version": "v1"})
    resp = await client.get("/models/rb-model/active")
    assert resp.json()["version"] == "v1"


@pytest.mark.asyncio
async def test_list_versions(client: AsyncClient, tmp_path) -> None:
    """List versions returns active first."""
    for v in ["v1", "v2", "v3"]:
        artifact_uri = create_fake_artifact(tmp_path, f"{v}.joblib")
        await client.post(
            "/models/register",
            json={"name": "list-model", "version": v, "artifact_uri": artifact_uri},
        )
    await client.post("/models/list-model/activate", json={"version": "v2"})

    resp = await client.get("/models/list-model")
    assert resp.status_code == 200
    versions = resp.json()
    assert len(versions) == 3
    assert versions[0]["version"] == "v2"
    assert versions[0]["is_active"] is True

@pytest.mark.asyncio
async def test_register_model_with_valid_artifact(client, tmp_path):
    """Test registering a model with a valid artifact file."""
    artifact_content = b"fake model artifact"
    artifact_uri = create_fake_artifact(tmp_path, "model.joblib", artifact_content)

    expected_hash = hashlib.sha256(artifact_content).hexdigest()

    response = await client.post(
        "/models/register",
        json={
            "name": "demo_classifier",
            "version": "v1",
            "artifact_uri": artifact_uri,
            "framework": "sklearn",
            "task_type": "classification",
            "tags": ["demo", "baseline"],
        },
    )

    assert response.status_code in (200, 201)
    data = response.json()

    assert data["name"] == "demo_classifier"
    assert data["version"] == "v1"
    assert data["artifact_uri"] == artifact_uri
    assert data["framework"] == "sklearn"
    assert data["task_type"] == "classification"
    assert data["tags"] == ["demo", "baseline"]
    assert data["status"] == "registered"
    assert data["artifact_hash"] == expected_hash

@pytest.mark.asyncio
async def test_register_model_rejects_missing_artifact(client, tmp_path):
    """Test that registering a model with a missing artifact file is rejected."""
    missing_path = tmp_path / "missing_model.joblib"

    response = await client.post(
        "/models/register",
        json={
            "name": "bad_model",
            "version": "v1",
            "artifact_uri": str(missing_path),
            "framework": "sklearn",
            "task_type": "classification",
            "tags": ["demo"],
        },
    )

    assert response.status_code == 422
    assert "Artifact not found" in response.json()["detail"]