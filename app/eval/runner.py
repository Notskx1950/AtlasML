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

        dataset = _load_dataset(dataset_path)
        inputs = [row["input"] for row in dataset]
        labels = [row["label"] for row in dataset]

        all_predictions: list[dict] = []
        latencies: list[float] = []

        # Run in batches
        for i in range(0, len(inputs), EvalRunner.BATCH_SIZE):
            batch = inputs[i : i + EvalRunner.BATCH_SIZE]
            start = time.perf_counter()
            batch_preds = asyncio.get_event_loop().run_until_complete(
                adapter.predict(batch)
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies.extend([elapsed / len(batch)] * len(batch))
            all_predictions.extend(batch_preds)

        # Determine metric type based on adapter
        is_llm = hasattr(adapter, "last_input_tokens")

        if is_llm:
            schema_flags = [adapter.schema_validate(p) for p in all_predictions]
            token_counts = [len(str(p.get("response", ""))) for p in all_predictions]
            metrics = compute_llm_metrics(latencies, schema_flags, token_counts)
        else:
            pred_values = [p.get("prediction") for p in all_predictions]
            metrics = compute_classification_metrics(labels, pred_values)

        # Write metrics to DB
        run_uuid = uuid.UUID(run_id)
        for name, value in metrics.items():
            db.add(EvalMetric(run_id=run_uuid, metric_name=name, value=value))

        # Update run status
        run = db.get(EvalRun, run_uuid)
        if run:
            run.status = "completed"
            run.finished_at = datetime.now(timezone.utc)

        db.commit()
        return metrics


def _load_dataset(path: str) -> list[dict]:
    """Load a newline-delimited JSON dataset."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
