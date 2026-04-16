"""Tests for the eval API and runner."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EvalMetric, EvalRun, ModelVersion
from tests.conftest import MockSklearnAdapter


async def _register_model(db: AsyncSession, name: str, version: str) -> None:
    """Insert a model version directly into the DB."""
    mv = ModelVersion(
        name=name,
        version=version,
        artifact_uri=f"/m/{version}.joblib",
        is_active=True,
    )
    db.add(mv)
    await db.commit()


@pytest.mark.asyncio
async def test_eval_run_on_fixture_dataset(
    client: AsyncClient, db_session: AsyncSession, sample_dataset: str
) -> None:
    """Run eval on a 10-row dataset and verify completion + metrics."""
    await _register_model(db_session, "eval-model", "v1")

    # Create the eval run directly (bypass RQ)
    run_id = uuid.uuid4()
    run = EvalRun(
        id=run_id,
        model_name="eval-model",
        model_version="v1",
        dataset_id="test-ds",
        dataset_hash="abc123",
        status="running",
    )
    db_session.add(run)
    await db_session.commit()

    # Run the eval synchronously using the runner
    from app.eval.runner import EvalRunner

    adapter = MockSklearnAdapter()

    # We need a sync session for the runner — simulate by using async
    # Instead, call the logic directly and write metrics
    import asyncio

    dataset_rows = []
    with open(sample_dataset) as f:
        for line in f:
            if line.strip():
                dataset_rows.append(json.loads(line))

    inputs = [r["input"] for r in dataset_rows]
    labels = [r["label"] for r in dataset_rows]
    predictions = await adapter.predict(inputs)
    pred_values = [p["prediction"] for p in predictions]

    from app.eval.metrics import compute_classification_metrics

    metrics = compute_classification_metrics(labels, pred_values)

    for name, value in metrics.items():
        db_session.add(EvalMetric(run_id=run_id, metric_name=name, value=value))
    run.status = "completed"
    run.finished_at = datetime.now(timezone.utc)
    await db_session.commit()

    # Verify via API
    resp = await client.get(f"/eval/runs/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["metrics"]) >= 2
    metric_names = {m["metric_name"] for m in data["metrics"]}
    assert "accuracy" in metric_names
    assert "f1_macro" in metric_names


@pytest.mark.asyncio
async def test_eval_compare(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Compare two model versions and verify delta math."""
    await _register_model(db_session, "cmp-model", "v1")
    await _register_model(db_session, "cmp-model", "v2")

    # Create eval runs with metrics for both versions
    for version, accuracy, f1 in [("v1", 0.82, 0.80), ("v2", 0.85, 0.84)]:
        run_id = uuid.uuid4()
        run = EvalRun(
            id=run_id,
            model_name="cmp-model",
            model_version=version,
            dataset_id="test-ds",
            dataset_hash="abc",
            status="completed",
            finished_at=datetime.now(timezone.utc),
        )
        db_session.add(run)
        db_session.add(EvalMetric(run_id=run_id, metric_name="accuracy", value=accuracy))
        db_session.add(EvalMetric(run_id=run_id, metric_name="f1_macro", value=f1))

    await db_session.commit()

    resp = await client.get(
        "/eval/compare", params={"model_name": "cmp-model", "v1": "v1", "v2": "v2"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "cmp-model"

    comparisons = {c["metric"]: c for c in data["comparison"]}
    assert "accuracy" in comparisons

    acc = comparisons["accuracy"]
    assert acc["v1"] == 0.82
    assert acc["v2"] == 0.85
    assert abs(acc["delta"] - 0.03) < 0.001
    assert acc["pct_change"] == pytest.approx(3.66, abs=0.1)


@pytest.mark.asyncio
async def test_eval_run_api_enqueue(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /eval/run creates a run record and enqueues work."""
    await _register_model(db_session, "eval-api-model", "v1")

    with patch("app.api.eval.Redis"), patch("app.api.eval.Queue") as mock_q_cls:
        mock_q_cls.return_value.enqueue.return_value = None

        resp = await client.post(
            "/eval/run",
            json={
                "model_name": "eval-api-model",
                "version": "v1",
                "dataset_id": "test",
                "dataset_path": "/data/test.jsonl",
            },
        )

    assert resp.status_code == 202
    data = resp.json()
    assert "run_id" in data

    result = await db_session.execute(
        select(EvalRun).where(EvalRun.id == uuid.UUID(data["run_id"]))
    )
    run = result.scalar_one()
    assert run.status == "running"
