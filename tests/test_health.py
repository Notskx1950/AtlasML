"""Test health check endpoint."""

from __future__ import annotations

from unittest.mock import patch
import pytest


@pytest.mark.asyncio
async def test_health_check(client) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_ready_returns_200_when_dependencies_ok(client):
    with patch("app.api.health.Redis") as mock_redis:
        mock_redis.from_url.return_value.ping.return_value = True

        resp = await client.get("/ready")

    assert resp.status_code == 200
    assert resp.json() == {
        "api": "ok",
        "postgres": "ok",
        "redis": "ok",
    }

@pytest.mark.asyncio
async def test_ready_returns_503_when_redis_fails(client):
    with patch("app.api.health.Redis") as mock_redis:
        mock_redis.from_url.return_value.ping.side_effect = RuntimeError("redis down")

        resp = await client.get("/ready")

    assert resp.status_code == 503
    assert resp.json()["api"] == "ok"
    assert resp.json()["postgres"] == "ok"
    assert resp.json()["redis"] == "error"