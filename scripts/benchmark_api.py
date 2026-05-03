# scripts/benchmark_api.py

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class BenchmarkResult:
    scenario: str
    runs: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * pct)
    return sorted_values[index]


def measure(
    scenario: str,
    runs: int,
    fn: Callable[[], httpx.Response],
) -> BenchmarkResult:
    latencies: list[float] = []

    for _ in range(runs):
        start = time.perf_counter()
        response = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code >= 400:
            raise RuntimeError(
                f"{scenario} failed with {response.status_code}: {response.text}"
            )

        latencies.append(elapsed_ms)

    return BenchmarkResult(
        scenario=scenario,
        runs=runs,
        avg_ms=statistics.mean(latencies),
        p50_ms=percentile(latencies, 0.50),
        p95_ms=percentile(latencies, 0.95),
        min_ms=min(latencies),
        max_ms=max(latencies),
    )


def print_markdown(results: list[BenchmarkResult]) -> None:
    print("| Scenario | Runs | Avg ms | P50 ms | P95 ms | Min ms | Max ms |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    for result in results:
        print(
            f"| {result.scenario} "
            f"| {result.runs} "
            f"| {result.avg_ms:.2f} "
            f"| {result.p50_ms:.2f} "
            f"| {result.p95_ms:.2f} "
            f"| {result.min_ms:.2f} "
            f"| {result.max_ms:.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default="demo-classifier")
    parser.add_argument("--dataset-id", default="benchmark-dataset")
    parser.add_argument("--dataset-path", default="data/eval_dataset.jsonl")
    args = parser.parse_args()

    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    client = httpx.Client(base_url=args.base_url, timeout=30.0, headers=headers)

    predict_payload: dict[str, Any] = {
        "model_name": args.model_name,
        "inputs": [
        {
            "feature_0": 5.1,
            "feature_1": 3.5,
            "feature_2": 1.4,
            "feature_3": 0.2,
        }
    ],
    }

    eval_payload: dict[str, Any] = {
        "model_name": args.model_name,
        "version": "v1",
        "dataset_id": args.dataset_id,
        "dataset_path": args.dataset_path,
        "config_snapshot": {"benchmark": True},
    }

    results = [
        measure("health check", args.runs, lambda: client.get("/health")),
        measure("readiness check", args.runs, lambda: client.get("/ready")),
        measure("sync predict", args.runs, lambda: client.post("/predict", json=predict_payload)),
        measure("async predict enqueue", args.runs, lambda: client.post("/jobs/predict", json=predict_payload)),
        measure("eval enqueue", args.runs, lambda: client.post("/eval/run", json=eval_payload)),
        measure("model stats query", args.runs, lambda: client.get(f"/models/{args.model_name}/stats")),
    ]

    print_markdown(results)


if __name__ == "__main__":
    main()