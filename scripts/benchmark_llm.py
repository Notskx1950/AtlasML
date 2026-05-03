# scripts/benchmark_llm.py

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any
from unittest import result

import httpx


@dataclass
class LLMBenchmarkResult:
    scenario: str
    runs: int
    success_rate: float
    schema_valid_rate: float | None
    avg_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_total_tokens: float | None = None


SCENARIOS: dict[str, dict[str, Any]] = {
    "short_prompt": {
        "inputs": [
            {"prompt": "Explain model latency in one sentence."}
        ],
        "schema_check": False,
    },
    "long_prompt": {
        "inputs": [
            {
                "prompt": (
                    "You are evaluating an AI infrastructure backend. "
                    "Explain the difference between synchronous inference, "
                    "asynchronous job enqueueing, evaluation runs, and model "
                    "observability metrics. Keep the answer under 150 words."
                )
            }
        ],
        "schema_check": False,
    },
    "json_output": {
        "inputs": [
            {
                "prompt": (
                    "Return only valid JSON with keys: "
                    "summary, risk_level, next_action. "
                    "The topic is: model registry rollback after failed eval."
                )
            }
        ],
        "schema_check": True,
    },
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * pct)
    return sorted_values[index]


def is_valid_json_response(response_text: str) -> bool:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return False

    required = {"summary", "risk_level", "next_action"}
    return isinstance(parsed, dict) and required.issubset(parsed.keys())


def benchmark_scenario(
    client: httpx.Client,
    model_name: str,
    scenario_name: str,
    scenario: dict[str, Any],
    runs: int,
) -> LLMBenchmarkResult:
    latencies: list[float] = []
    successes = 0
    schema_valid_count = 0
    schema_checks = 0

    input_tokens_list: list[int] = []
    output_tokens_list: list[int] = []
    total_tokens_list: list[int] = []

    for _ in range(runs):
        payload = {
            "model_name": model_name,
            "inputs": scenario["inputs"],
        }

        start = time.perf_counter()
        response = client.post("/predict", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)

        if response.status_code < 400:
            successes += 1

            body = response.json()
            predictions = body.get("predictions", [])

            if scenario.get("schema_check"):
                schema_checks += 1
                if predictions:
                    raw = predictions[0].get("response", "")
                    if is_valid_json_response(raw):
                        schema_valid_count += 1
    token_usage = body.get("token_usage")
    if token_usage:
        if token_usage.get("input_tokens") is not None:
            input_tokens_list.append(token_usage["input_tokens"])
        if token_usage.get("output_tokens") is not None:
            output_tokens_list.append(token_usage["output_tokens"])
        if token_usage.get("total_tokens") is not None:
            total_tokens_list.append(token_usage["total_tokens"])

    def avg_or_none(values: list[int]) -> float | None:
        return sum(values) / len(values) if values else None
        
    success_rate = successes / runs if runs else 0.0
    schema_valid_rate = (
        schema_valid_count / schema_checks if schema_checks else None
    )

    return LLMBenchmarkResult(
        scenario=scenario_name,
        runs=runs,
        success_rate=success_rate,
        schema_valid_rate=schema_valid_rate,
        avg_ms=statistics.mean(latencies),
        p50_ms=percentile(latencies, 0.50),
        p95_ms=percentile(latencies, 0.95),
        min_ms=min(latencies),
        max_ms=max(latencies),
        avg_input_tokens=avg_or_none(input_tokens_list),
        avg_output_tokens=avg_or_none(output_tokens_list),
        avg_total_tokens=avg_or_none(total_tokens_list),
        )


def print_markdown(results: list[LLMBenchmarkResult]) -> None:
    print(
        "| Scenario | Runs | Success Rate | Schema Valid Rate | "
        "Avg ms | P50 ms | P95 ms | Avg Input Tokens | Avg Output Tokens | Avg Total Tokens |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for result in results:
        schema_rate = (
            f"{result.schema_valid_rate:.2f}"
            if result.schema_valid_rate is not None
            else "N/A"
        )
        avg_input = (
            f"{result.avg_input_tokens:.2f}"
            if result.avg_input_tokens is not None
            else "N/A"
        )
        avg_output = (
            f"{result.avg_output_tokens:.2f}"
            if result.avg_output_tokens is not None
            else "N/A"
        )
        avg_total = (
            f"{result.avg_total_tokens:.2f}"
            if result.avg_total_tokens is not None
            else "N/A"
        )

        print(
            f"| {result.scenario} "
            f"| {result.runs} "
            f"| {result.success_rate:.2f} "
            f"| {schema_rate} "
            f"| {result.avg_ms:.2f} "
            f"| {result.p50_ms:.2f} "
            f"| {result.p95_ms:.2f} "
            f"| {avg_input} "
            f"| {avg_output} "
            f"| {avg_total} |"
        )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="llm-benchmark")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    client = httpx.Client(
        base_url=args.base_url,
        timeout=120.0,
        headers=headers,
    )

    results = [
        benchmark_scenario(
            client=client,
            model_name=args.model_name,
            scenario_name=name,
            scenario=scenario,
            runs=args.runs,
        )
        for name, scenario in SCENARIOS.items()
    ]

    print_markdown(results)


if __name__ == "__main__":
    main()