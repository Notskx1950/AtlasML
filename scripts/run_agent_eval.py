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
    category: str
    difficulty: str
    run_status: str
    task_success: bool
    tool_correct: bool
    controlled_failure: bool | None
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


def is_failure_case(category: str) -> bool:
    return category == "failure_cases"


def failure_case_handled(run: dict) -> bool:
    return (
        run.get("status") == "failed"
        and run.get("error_type") is not None
        and run.get("error_message") is not None
    )


def _mean(values: list[float | int]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize_results(results: list[AgentEvalResult]) -> dict[str, float | int]:
    total = len(results)
    normal_results = [r for r in results if not is_failure_case(r.category)]
    failure_results = [r for r in results if is_failure_case(r.category)]

    controlled_failures = [
        r.controlled_failure
        for r in failure_results
        if r.controlled_failure is not None
    ]
    durations = [r.duration_ms for r in results if r.duration_ms is not None]
    tokens = [r.total_tokens for r in results if r.total_tokens is not None]
    steps = [r.step_count for r in results]

    normal_task_success = (
        sum(r.task_success for r in normal_results) / len(normal_results)
        if normal_results
        else 0.0
    )
    tool_accuracy = sum(r.tool_correct for r in results) / total if total else 0.0
    controlled_failure_rate = (
        sum(controlled_failures) / len(controlled_failures)
        if controlled_failures
        else 0.0
    )

    return {
        "runs": total,
        "normal_task_success": normal_task_success,
        "tool_accuracy": tool_accuracy,
        "controlled_failure_rate": controlled_failure_rate,
        "avg_steps": _mean(steps),
        "avg_duration_ms": _mean(durations),
        "avg_tokens": _mean(tokens),
    }


def print_summary(results: list[AgentEvalResult]) -> None:
    summary = summarize_results(results)

    print(
        "| Task Set | Runs | Normal Task Success | Tool Accuracy | "
        "Controlled Failure Rate | Avg Steps | Avg Latency ms | Avg Tokens |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| agent_eval_tasks "
        f"| {summary['runs']} "
        f"| {summary['normal_task_success']:.2f} "
        f"| {summary['tool_accuracy']:.2f} "
        f"| {summary['controlled_failure_rate']:.2f} "
        f"| {summary['avg_steps']:.2f} "
        f"| {summary['avg_duration_ms']:.2f} "
        f"| {summary['avg_tokens']:.2f} |"
    )

def print_category_summary(results: list[AgentEvalResult]) -> None:
    categories = sorted({r.category for r in results})

    print("\n## By Category")
    print(
        "| Category | Runs | Normal Task Success | Tool Accuracy | "
        "Controlled Failure Rate | Avg Steps | Avg Latency ms | Avg Tokens |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for category in categories:
        subset = [r for r in results if r.category == category]
        summary = summarize_results(subset)

        print(
            f"| {category} "
            f"| {summary['runs']} "
            f"| {summary['normal_task_success']:.2f} "
            f"| {summary['tool_accuracy']:.2f} "
            f"| {summary['controlled_failure_rate']:.2f} "
            f"| {summary['avg_steps']:.2f} "
            f"| {summary['avg_duration_ms']:.2f} "
            f"| {summary['avg_tokens']:.2f} |"
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

        category = item.get("category", "uncategorized")
        failure_case = is_failure_case(category)
        task_success = (
            False
            if failure_case
            else contains_expected_answer(
                run.get("final_output"),
                item.get("expected_answer_contains"),
            )
        )
        tool_correct = tool_called_correctly(
            trace,
            item.get("expected_tool"),
        )
        controlled_failure = failure_case_handled(run) if failure_case else None

        results.append(
            AgentEvalResult(
                task=item["task"],
                category=category,
                difficulty=item.get("difficulty", "unknown"),
                run_status=run.get("status", "unknown"),
                task_success=task_success,
                tool_correct=tool_correct,
                controlled_failure=controlled_failure,
                step_count=len(trace.get("steps", [])),
                duration_ms=run.get("duration_ms"),
                total_tokens=run.get("total_tokens"),
            )
        )

    print_summary(results)
    print_category_summary(results)


if __name__ == "__main__":
    main()
