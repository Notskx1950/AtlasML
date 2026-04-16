"""Tests for the inference API."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import AsyncClient

from app.models.registry_store import RegistryStore
from tests.conftest import BrokenAdapter, MockSklearnAdapter, SchemaFailAdapter


async def _register_and_activate(client: AsyncClient, name: str, version: str) -> None:
    """Helper to register and activate a model."""
    await client.post(
        "/models/register",
        json={"name": name, "version": version, "artifact_uri": f"/m/{version}.joblib"},
    )
    await client.post(f"/models/{name}/activate", json={"version": version})


@pytest.mark.asyncio
async def test_sync_predict(client: AsyncClient, db_session) -> None:
    """Sync predict with a mock adapter returns correct shape."""
    await _register_and_activate(client, "pred-model", "v1")

    # Inject mock adapter into the store
    store = RegistryStore()
    store.put("pred-model", "v1", MockSklearnAdapter())

    resp = await client.post(
        "/predict",
        json={"model_name": "pred-model", "inputs": [{"x": 1}, {"x": 2}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2
    assert data["model_version"] == "v1"
    assert "latency_ms" in data

    # Verify inference log was written
    from sqlalchemy import select
    from app.db.models import InferenceLog

    result = await db_session.execute(
        select(InferenceLog).where(InferenceLog.model_name == "pred-model")
    )
    log = result.scalar_one()
    assert log.status_code == 200
    assert log.latency_ms > 0

    # Cleanup singleton state
    store.evict("pred-model", "v1")


@pytest.mark.asyncio
async def test_sync_predict_broken_model(client: AsyncClient, db_session) -> None:
    """Predict with a broken adapter returns error and logs it."""
    await _register_and_activate(client, "broken-model", "v1")

    store = RegistryStore()
    store.put("broken-model", "v1", BrokenAdapter())

    resp = await client.post(
        "/predict",
        json={"model_name": "broken-model", "inputs": [{"x": 1}]},
    )
    assert resp.status_code == 422

    from sqlalchemy import select
    from app.db.models import InferenceLog

    result = await db_session.execute(
        select(InferenceLog).where(InferenceLog.model_name == "broken-model")
    )
    log = result.scalar_one()
    assert log.status_code == 422
    assert log.error_type == "FileNotFoundError"

    store.evict("broken-model", "v1")


@pytest.mark.asyncio
async def test_sync_predict_schema_fail(client: AsyncClient, db_session) -> None:
    """Predict with schema validation failure sets schema_valid=False."""
    await _register_and_activate(client, "schema-model", "v1")

    store = RegistryStore()
    store.put("schema-model", "v1", SchemaFailAdapter())

    resp = await client.post(
        "/predict",
        json={"model_name": "schema-model", "inputs": [{"x": 1}]},
    )
    assert resp.status_code == 200

    from sqlalchemy import select
    from app.db.models import InferenceLog

    result = await db_session.execute(
        select(InferenceLog).where(InferenceLog.model_name == "schema-model")
    )
    log = result.scalar_one()
    assert log.schema_valid is False

    store.evict("schema-model", "v1")


@pytest.mark.asyncio
async def test_async_predict_enqueues_job(client: AsyncClient, db_session) -> None:
    """Async predict returns 202 and creates a job record."""
    await _register_and_activate(client, "async-model", "v1")

    # Mock Redis/RQ so we don't need a real connection
    with patch("app.api.inference.Redis") as mock_redis_cls, \
         patch("app.api.inference.Queue") as mock_queue_cls:
        mock_queue = mock_queue_cls.return_value
        mock_queue.enqueue.return_value = None

        resp = await client.post(
            "/jobs/predict",
            json={"model_name": "async-model", "inputs": [{"x": 1}]},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data

    # Verify job record exists
    from sqlalchemy import select
    from app.db.models import JobRecord
    import uuid

    result = await db_session.execute(
        select(JobRecord).where(JobRecord.id == uuid.UUID(data["job_id"]))
    )
    record = result.scalar_one()
    assert record.status == "queued"
    assert record.model_name == "async-model"
