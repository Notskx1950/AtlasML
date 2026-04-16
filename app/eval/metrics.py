"""Metric computation functions for eval runs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    labels: list, predictions: list
) -> dict[str, float]:
    """Compute accuracy and F1 scores for classification tasks."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def compute_llm_metrics(
    latencies_ms: list[float],
    schema_valid_flags: list[bool],
    output_token_counts: list[int],
) -> dict[str, float]:
    """Compute LLM-specific metrics."""
    arr = np.array(latencies_ms)
    metrics: dict[str, float] = {
        "p50_latency_ms": float(np.percentile(arr, 50)) if len(arr) > 0 else 0.0,
        "p95_latency_ms": float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0,
        "format_success_rate": (
            sum(schema_valid_flags) / len(schema_valid_flags)
            if schema_valid_flags
            else 0.0
        ),
    }
    if output_token_counts:
        metrics["avg_output_tokens"] = float(np.mean(output_token_counts))
    return metrics
