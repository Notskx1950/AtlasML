"""Tests for the model registry API."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_model(client: AsyncClient) -> None:
    """Register a model version and verify the response."""
    resp = await client.post(
        "/models/register",
        json={
            "name": "test-model",
            "version": "v1.0",
            "artifact_uri": "/models/test.joblib",
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
async def test_activate_model(client: AsyncClient) -> None:
    """Activate a version and verify only it is active."""
    # Register two versions
    await client.post(
        "/models/register",
        json={"name": "act-model", "version": "v1", "artifact_uri": "/m/v1.joblib"},
    )
    await client.post(
        "/models/register",
        json={"name": "act-model", "version": "v2", "artifact_uri": "/m/v2.joblib"},
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
async def test_rollback_activation(client: AsyncClient) -> None:
    """Activate v1, then v2, then rollback to v1."""
    for v in ["v1", "v2"]:
        await client.post(
            "/models/register",
            json={"name": "rb-model", "version": v, "artifact_uri": f"/m/{v}.joblib"},
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
async def test_list_versions(client: AsyncClient) -> None:
    """List versions returns active first."""
    for v in ["v1", "v2", "v3"]:
        await client.post(
            "/models/register",
            json={"name": "list-model", "version": v, "artifact_uri": f"/m/{v}.joblib"},
        )
    await client.post("/models/list-model/activate", json={"version": "v2"})

    resp = await client.get("/models/list-model")
    assert resp.status_code == 200
    versions = resp.json()
    assert len(versions) == 3
    assert versions[0]["version"] == "v2"
    assert versions[0]["is_active"] is True
