from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class AgentBenchmarkResult:
    scenario: str
    runs: int
    success_rate: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    avg_steps: float | None
    avg_tokens: float | None

SCENARIOS = {
    "calculator_agent_run": "Calculate 18 * 23 and explain the result.",
    "echo_json_agent_run": 'Echo this JSON: {"status":"ok"}.',
    "model_stats_lookup_agent_run": "Look up stats for model llm-benchmark.",
}


def percentile(values: list[float], pct: float) -> float:
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * pct)
    return sorted_values[index]


def print_markdown(results: list[AgentBenchmarkResult]) -> None:
    print("| Scenario | Runs | Success Rate | Avg ms | P50 ms | P95 ms | Avg Steps | Avg Tokens |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for result in results:
        print(
            f"| {result.scenario} "
            f"| {result.runs} "
            f"| {result.success_rate:.2f} "
            f"| {result.avg_ms:.2f} "
            f"| {result.p50_ms:.2f} "
            f"| {result.p95_ms:.2f} "
            f"| {result.avg_steps if result.avg_steps is not None else 'N/A'} "
            f"| {result.avg_tokens if result.avg_tokens is not None else 'N/A'} |"
        )


def benchmark_agent_run(
    client: httpx.Client,
    model_name: str,
    task: str,
    scenario_name: str,
    runs: int,
) -> AgentBenchmarkResult:
    latencies: list[float] = []
    successes = 0
    step_counts: list[int] = []
    token_counts: list[int] = []

    for _ in range(runs):
        start = time.perf_counter()
        response = client.post(
            "/agents/run",
            json={
                "model_name": model_name,
                "task": task,
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        if response.status_code < 400:
            run = response.json()
            if run.get("status") == "completed":
                successes += 1

            if run.get("total_tokens") is not None:
                token_counts.append(run["total_tokens"])

            trace = client.get(f"/agents/runs/{run['id']}/trace").json()
            step_counts.append(len(trace.get("steps", [])))

    return AgentBenchmarkResult(
        scenario=scenario_name,
        runs=runs,
        success_rate=successes / runs if runs else 0.0,
        avg_ms=statistics.mean(latencies),
        p50_ms=percentile(latencies, 0.50),
        p95_ms=percentile(latencies, 0.95),
        avg_steps=statistics.mean(step_counts) if step_counts else None,
        avg_tokens=statistics.mean(token_counts) if token_counts else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default="llm-benchmark")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    client = httpx.Client(
        base_url=args.base_url,
        timeout=120.0,
        headers=headers,
    )

    results = [
        benchmark_agent_run(
            client=client,
            model_name=args.model_name,
            task=task,
            scenario_name=name,
            runs=args.runs,
        )
        for name, task in SCENARIOS.items()
    ]

    print_markdown(results)


if __name__ == "__main__":
    main()