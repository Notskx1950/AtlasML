from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_write_endpoint_requires_api_key_when_enabled(
    client: AsyncClient,
) -> None:
    with patch("app.api.deps.settings.API_KEY", "dev-secret-key"):
        resp = await client.post(
            "/models/register",
            json={
                "name": "auth-model",
                "version": "v1",
                "artifact_uri": "/tmp/missing.joblib",
            },
        )

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid API key"

@pytest.mark.asyncio
async def test_write_endpoint_rejects_wrong_api_key_when_enabled(
    client: AsyncClient,
) -> None:
    with patch("app.api.deps.settings.API_KEY", "dev-secret-key"):
        resp = await client.post(
            "/models/register",
            headers={"X-API-Key": "wrong-key"},
            json={
                "name": "auth-model",
                "version": "v1",
                "artifact_uri": "/tmp/missing.joblib",
            },
        )

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid API key"

@pytest.mark.asyncio
async def test_write_endpoint_accepts_correct_api_key_when_enabled(
    client: AsyncClient,
) -> None:
    with patch("app.api.deps.settings.API_KEY", "dev-secret-key"):
        resp = await client.post(
            "/models/register",
            headers={"X-API-Key": "dev-secret-key"},
            json={
                "name": "auth-model",
                "version": "v1",
                "artifact_uri": "/tmp/missing.joblib",
            },
        )

    assert resp.status_code == 400
    assert "Artifact not found" in resp.json()["detail"]

@pytest.mark.asyncio
async def test_write_endpoint_allows_request_when_api_key_unset(
    client: AsyncClient,
) -> None:
    with patch("app.api.deps.settings.API_KEY", None):
        resp = await client.post(
            "/models/register",
            json={
                "name": "auth-disabled-model",
                "version": "v1",
                "artifact_uri": "/tmp/missing.joblib",
            },
        )

    assert resp.status_code == 400
    assert "Artifact not found" in resp.json()["detail"]

@pytest.mark.asyncio
async def test_read_endpoint_does_not_require_api_key_when_enabled(
    client: AsyncClient,
) -> None:
    with patch("app.api.deps.settings.API_KEY", "dev-secret-key"):
        resp = await client.get("/health")

    assert resp.status_code == 200