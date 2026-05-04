"""Tests for the model statistics API."""

from __future__ import annotations


import pytest

from app.db.models import ModelVersion, InferenceLog

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

@pytest.mark.asyncio
async def test_model_stats_aggregates_token_metrics(client, db_session):
    model = ModelVersion(
        name="llm-stats-model",
        version="v1",
        artifact_uri="gpt-4o-mini",
        runtime_config={"adapter_type": "llm"},
        is_active=True,
        status="registered",
    )
    db_session.add(model)

    db_session.add(
        InferenceLog(
            model_name="llm-stats-model",
            model_version="v1",
            latency_ms=100.0,
            status_code=200,
            input_tokens=10,
            output_tokens=20,
        )
    )
    db_session.add(
        InferenceLog(
            model_name="llm-stats-model",
            model_version="v1",
            latency_ms=200.0,
            status_code=200,
            input_tokens=30,
            output_tokens=40,
        )
    )
    await db_session.commit()

    resp = await client.get("/models/llm-stats-model/stats")

    assert resp.status_code == 200
    data = resp.json()

    assert data["total_input_tokens"] == 40
    assert data["total_output_tokens"] == 60
    assert data["total_tokens"] == 100
    assert data["avg_input_tokens"] == 20.0
    assert data["avg_output_tokens"] == 30.0
    assert data["avg_total_tokens"] == 50.0

@pytest.mark.asyncio
async def test_model_stats_without_token_logs_returns_zero_token_totals(client, db_session):
    model = ModelVersion(
        name="sklearn-stats-model",
        version="v1",
        artifact_uri="/tmp/model.joblib",
        is_active=True,
        status="registered",
    )
    db_session.add(model)

    db_session.add(
        InferenceLog(
            model_name="sklearn-stats-model",
            model_version="v1",
            latency_ms=10.0,
            status_code=200,
        )
    )
    await db_session.commit()

    resp = await client.get("/models/sklearn-stats-model/stats")

    assert resp.status_code == 200
    data = resp.json()

    assert data["total_input_tokens"] == 0
    assert data["total_output_tokens"] == 0
    assert data["total_tokens"] == 0
    assert data["avg_input_tokens"] is None
    assert data["avg_output_tokens"] is None
    assert data["avg_total_tokens"] is None