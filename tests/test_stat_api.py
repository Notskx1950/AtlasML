"""Tests for the model statistics API."""

from __future__ import annotations

import hashlib

import pytest
from httpx import AsyncClient

from app.db.models import ModelVersion, DeploymentEvent, InferenceLog

@pytest.mark.asyncio
async def test_model_stats_aggregates_inference_logs(client, db_session):
    model = ModelVersion(
        name="stats-model",
        version="v1",
        artifact_uri="/tmp/model.joblib",
        is_active=True,
        status="registered",
    )
    db_session.add(model)

    db_session.add(
        InferenceLog(
            model_name="stats-model",
            model_version="v1",
            latency_ms=10.0,
            status_code=200,
            error_type=None,
        )
    )
    db_session.add(
        InferenceLog(
            model_name="stats-model",
            model_version="v1",
            latency_ms=20.0,
            status_code=200,
            error_type=None,
        )
    )
    db_session.add(
        InferenceLog(
            model_name="stats-model",
            model_version="v1",
            latency_ms=30.0,
            status_code=500,
            error_type="PredictionError",
        )
    )

    await db_session.commit()

    resp = await client.get("/models/stats-model/stats")

    assert resp.status_code == 200
    data = resp.json()

    assert data["model_name"] == "stats-model"
    assert data["active_version"] == "v1"
    assert data["total_predictions"] == 3
    assert data["success_count"] == 2
    assert data["error_count"] == 1
    assert data["error_rate"] == pytest.approx(0.3333, abs=0.001)
    assert data["avg_latency_ms"] == 20.0
    assert data["top_error_types"] == {"PredictionError": 1}