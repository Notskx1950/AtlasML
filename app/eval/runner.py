"""Eval orchestration — runs a dataset through a model and records metrics."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import EvalMetric, EvalRun
from app.eval.metrics import compute_classification_metrics, compute_llm_metrics
from app.models.base import ModelAdapter
from app.utils.hashing import file_sha256


class EvalRunner:
    """Orchestrates evaluation of a model against a dataset."""

    BATCH_SIZE = 32

    @staticmethod
    def run_sync(
        run_id: str,
        adapter: ModelAdapter,
        dataset_path: str,
        db: Session,
    ) -> dict[str, float]:
        """Execute eval synchronously (called from RQ worker)."""
        import asyncio
        import uuid
        # run_id is a string when passed via RQ, but we want to store it as a UUID in the DB
        run_uuid = uuid.UUID(run_id)
        started_at = datetime.now(timezone.utc)
        # Update the EvalRun record to "running" and set start time
        run = db.get(EvalRun, run_uuid)
        if run is None:
            raise ValueError(f"EvalRun {run_id} not found")
        
        run.status = "running"
        run.started_at = started_at
        run.dataset_path = str(dataset_path)
        db.commit()
        try:
            # Load dataset and compute hash and row count for tracking
            dataset = _load_dataset(dataset_path)
            dataset_hash = file_sha256(dataset_path)
            row_count = len(dataset)
            if row_count == 0:
                raise ValueError(f"Eval dataset is empty: {dataset_path}")
            # Update EvalRun with dataset info
            if run:
                run.dataset_hash = dataset_hash
                run.row_count = row_count
                db.commit()
            # Prepare inputs and labels for evaluation
            inputs = [row["input"] for row in dataset]
            labels = [row["label"] for row in dataset]
            # Run predictions in batches and collect latencies
            all_predictions: list[dict] = []
            latencies: list[float] = []

            for i in range(0, len(inputs), EvalRunner.BATCH_SIZE):
                batch = inputs[i : i + EvalRunner.BATCH_SIZE]
                start = time.perf_counter()
                batch_preds = asyncio.get_event_loop().run_until_complete(
                    adapter.predict(batch)
                )
                elapsed = (time.perf_counter() - start) * 1000
                latencies.extend([elapsed / len(batch)] * len(batch))
                all_predictions.extend(batch_preds)
            # Determine metric type based on adapter capabilities.
            is_llm = hasattr(adapter, "last_input_tokens")

            if is_llm:
                schema_flags = [adapter.schema_validate(p) for p in all_predictions]
                token_counts = [len(str(p.get("response", ""))) for p in all_predictions]
                metrics = compute_llm_metrics(latencies, schema_flags, token_counts)
            else:
                pred_values = [p.get("prediction") for p in all_predictions]
                metrics = compute_classification_metrics(labels, pred_values)

            for name, value in metrics.items():
                db.add(EvalMetric(run_id=run_uuid, metric_name=name, value=value))

            _mark_run_finished(run, "completed", started_at, db)
            db.commit()
            return metrics

        except Exception:
            _mark_run_finished(run, "failed", started_at, db)
            db.commit()
            raise


def _load_dataset(path: str | Path) -> list[dict]:
    """Load a newline-delimited JSON dataset."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _duration_ms(started_at: datetime, finished_at: datetime) -> int:
    return int((finished_at - started_at).total_seconds() * 1000)


def _mark_run_finished(
    run: EvalRun | None,
    status: str,
    started_at: datetime,
    db: Session,
) -> None:
    if not run:
        return

    finished_at = datetime.now(timezone.utc)
    run.status = status
    run.finished_at = finished_at
    run.duration_ms = _duration_ms(started_at, finished_at)
