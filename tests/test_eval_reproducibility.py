"""Tests for eval reproducibility metadata."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, EvalMetric, EvalRun
from app.utils.hashing import file_sha256
from tests.conftest import MockSklearnAdapter
from tests.test_eval import _register_model


@pytest.mark.asyncio
async def test_eval_run_api_creates_reproducibility_shell(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """POST /eval/run creates an EvalRun shell before the worker executes."""
    await _register_model(db_session, "eval-api-model", "v1")

    with patch("app.api.eval.Redis"), patch("app.api.eval.Queue") as mock_q_cls:
        mock_q_cls.return_value.enqueue.return_value = None

        resp = await client.post(
            "/eval/run",
            json={
                "model_name": "eval-api-model",
                "version": "v1",
                "dataset_id": "test-ds",
                "dataset_path": "/data/test.jsonl",
                "git_commit": "abc123",
                "config_snapshot": {"batch_size": 32, "metric_set": ["accuracy"]},
            },
        )

    assert resp.status_code == 202

    body = resp.json()
    assert "run_id" in body
    assert body["status"] == "running"

    result = await db_session.execute(
        select(EvalRun).where(EvalRun.id == uuid.UUID(body["run_id"]))
    )
    run = result.scalar_one()

    assert run.model_name == "eval-api-model"
    assert run.model_version == "v1"
    assert run.dataset_id == "test-ds"
    assert run.dataset_path == "/data/test.jsonl"
    assert run.dataset_hash == "pending"
    assert run.row_count is None
    assert run.duration_ms is None
    assert run.status == "running"
    assert run.git_commit == "abc123"
    assert run.config_snapshot == {"batch_size": 32, "metric_set": ["accuracy"]}


@pytest.mark.asyncio
async def test_get_eval_run_returns_reproducibility_metadata(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """GET /eval/runs/{run_id} exposes eval reproducibility metadata."""
    run_id = uuid.uuid4()

    run = EvalRun(
        id=run_id,
        model_name="eval-model",
        model_version="v1",
        dataset_id="test-ds",
        dataset_path="/data/eval.jsonl",
        dataset_hash="a" * 64,
        row_count=10,
        git_commit="abc123",
        config_snapshot={"batch_size": 32, "metric_set": ["accuracy", "f1_macro"]},
        status="completed",
        finished_at=datetime.now(timezone.utc),
        duration_ms=123,
    )

    db_session.add(run)
    db_session.add(EvalMetric(run_id=run_id, metric_name="accuracy", value=0.9))
    db_session.add(EvalMetric(run_id=run_id, metric_name="f1_macro", value=0.88))
    await db_session.commit()

    resp = await client.get(f"/eval/runs/{run_id}")

    assert resp.status_code == 200

    body = resp.json()

    assert body["id"] == str(run_id)
    assert body["model_name"] == "eval-model"
    assert body["model_version"] == "v1"

    assert body["dataset_id"] == "test-ds"
    assert body["dataset_path"] == "/data/eval.jsonl"
    assert body["dataset_hash"] == "a" * 64
    assert body["row_count"] == 10

    assert body["git_commit"] == "abc123"
    assert body["config_snapshot"] == {
        "batch_size": 32,
        "metric_set": ["accuracy", "f1_macro"],
    }

    assert body["status"] == "completed"
    assert body["finished_at"] is not None
    assert body["duration_ms"] == 123

    metric_names = {metric["metric_name"] for metric in body["metrics"]}
    assert "accuracy" in metric_names
    assert "f1_macro" in metric_names


def test_eval_runner_records_reproducibility_metadata(sample_dataset: str) -> None:
    """EvalRunner fills actual dataset hash, row count, duration, and metrics."""
    from app.eval.runner import EvalRunner

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    db = SessionLocal()

    try:
        run_id = uuid.uuid4()

        run = EvalRun(
            id=run_id,
            model_name="eval-model",
            model_version="v1",
            dataset_id="test-ds",
            dataset_path=None,
            dataset_hash="pending",
            status="running",
        )

        db.add(run)
        db.commit()

        adapter = MockSklearnAdapter()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            metrics = EvalRunner.run_sync(
                run_id=str(run_id),
                adapter=adapter,
                dataset_path=sample_dataset,
                db=db,
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        updated = db.get(EvalRun, run_id)

        assert updated is not None
        assert updated.status == "completed"
        assert updated.dataset_path == sample_dataset
        assert updated.dataset_hash == file_sha256(sample_dataset)
        assert len(updated.dataset_hash) == 64
        assert updated.row_count == 10
        assert updated.finished_at is not None
        assert updated.duration_ms is not None
        assert updated.duration_ms >= 0

        assert "accuracy" in metrics
        assert "f1_macro" in metrics

        metric_rows = db.query(EvalMetric).filter(EvalMetric.run_id == run_id).all()
        metric_names = {metric.metric_name for metric in metric_rows}

        assert "accuracy" in metric_names
        assert "f1_macro" in metric_names

    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_eval_runner_marks_run_failed_on_missing_dataset() -> None:
    """EvalRunner marks the run as failed when dataset loading fails."""
    from app.eval.runner import EvalRunner

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    db = SessionLocal()

    try:
        run_id = uuid.uuid4()

        run = EvalRun(
            id=run_id,
            model_name="eval-model",
            model_version="v1",
            dataset_id="missing-ds",
            dataset_path=None,
            dataset_hash="pending",
            status="running",
        )

        db.add(run)
        db.commit()

        adapter = MockSklearnAdapter()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            with pytest.raises(FileNotFoundError):
                EvalRunner.run_sync(
                    run_id=str(run_id),
                    adapter=adapter,
                    dataset_path="/does/not/exist.jsonl",
                    db=db,
                )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        updated = db.get(EvalRun, run_id)

        assert updated is not None
        assert updated.status == "failed"
        assert updated.finished_at is not None
        assert updated.duration_ms is not None
        assert updated.duration_ms >= 0
        assert updated.dataset_hash == "pending"

    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()