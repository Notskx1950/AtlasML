from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class AgentEvalResult:
    task: str
    task_success: bool
    tool_correct: bool
    step_count: int
    duration_ms: int | None
    total_tokens: int | None


def load_tasks(path: str) -> list[dict]:
    tasks = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def contains_expected_answer(final_output: str | None, expected: str | None) -> bool:
    if not expected:
        return True
    if not final_output:
        return False
    return expected.lower() in final_output.lower()


def tool_called_correctly(trace: dict, expected_tool: str | None) -> bool:
    if not expected_tool:
        return True

    return any(
        invocation.get("tool_name") == expected_tool
        for invocation in trace.get("tool_invocations", [])
    )


def print_summary(results: list[AgentEvalResult]) -> None:
    total = len(results)

    task_success = sum(r.task_success for r in results) / total if total else 0
    tool_accuracy = sum(r.tool_correct for r in results) / total if total else 0

    durations = [r.duration_ms for r in results if r.duration_ms is not None]
    tokens = [r.total_tokens for r in results if r.total_tokens is not None]
    steps = [r.step_count for r in results]

    avg_duration = statistics.mean(durations) if durations else 0
    avg_tokens = statistics.mean(tokens) if tokens else 0
    avg_steps = statistics.mean(steps) if steps else 0

    print("| Task Set | Runs | Task Success | Tool Accuracy | Avg Steps | Avg Latency ms | Avg Tokens |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print(
        f"| agent_eval_tasks "
        f"| {total} "
        f"| {task_success:.2f} "
        f"| {tool_accuracy:.2f} "
        f"| {avg_steps:.2f} "
        f"| {avg_duration:.2f} "
        f"| {avg_tokens:.2f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--tasks", default="data/agent_eval_tasks.jsonl")
    parser.add_argument("--model-name", default="llm-benchmark")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    client = httpx.Client(base_url=args.base_url, timeout=120.0, headers=headers)

    tasks = load_tasks(args.tasks)
    results: list[AgentEvalResult] = []

    for item in tasks:
        response = client.post(
            "/agents/run",
            json={
                "task": item["task"],
                "model_name": args.model_name,
            },
        )
        response.raise_for_status()

        run = response.json()
        run_id = run["id"]

        trace_response = client.get(f"/agents/runs/{run_id}/trace")
        trace_response.raise_for_status()
        trace = trace_response.json()

        task_success = contains_expected_answer(
            run.get("final_output"),
            item.get("expected_answer_contains"),
        )
        tool_correct = tool_called_correctly(
            trace,
            item.get("expected_tool"),
        )

        results.append(
            AgentEvalResult(
                task=item["task"],
                task_success=task_success,
                tool_correct=tool_correct,
                step_count=len(trace.get("steps", [])),
                duration_ms=run.get("duration_ms"),
                total_tokens=run.get("total_tokens"),
            )
        )

    print_summary(results)


if __name__ == "__main__":
    main()