"""Tests for agent eval scoring helpers."""

from __future__ import annotations

from scripts.run_agent_eval import (
    AgentEvalResult,
    contains_expected_answer,
    failure_case_handled,
    summarize_results,
    tool_called_correctly,
)


def test_contains_expected_answer_returns_true_when_expected_text_present() -> None:
    assert contains_expected_answer("18 * 23 = 414", "414") is True


def test_contains_expected_answer_returns_false_when_expected_text_missing() -> None:
    assert contains_expected_answer("wrong answer", "414") is False


def test_contains_expected_answer_returns_true_when_no_expected_text() -> None:
    assert contains_expected_answer("anything", None) is True


def test_contains_expected_answer_returns_false_when_output_is_none() -> None:
    assert contains_expected_answer(None, "414") is False


def test_tool_called_correctly_returns_true_for_expected_tool() -> None:
    trace = {
        "tool_invocations": [
            {"tool_name": "calculator"},
        ]
    }

    assert tool_called_correctly(trace, "calculator") is True


def test_tool_called_correctly_returns_false_for_wrong_tool() -> None:
    trace = {
        "tool_invocations": [
            {"tool_name": "calculator"},
        ]
    }

    assert tool_called_correctly(trace, "model_stats_lookup") is False


def test_tool_called_correctly_returns_true_when_no_expected_tool() -> None:
    trace = {
        "tool_invocations": [
            {"tool_name": "calculator"},
        ]
    }

    assert tool_called_correctly(trace, None) is True


def test_tool_called_correctly_returns_false_when_no_tools_called() -> None:
    trace = {
        "tool_invocations": []
    }

    assert tool_called_correctly(trace, "calculator") is False


def test_failure_case_handled_requires_failed_run_with_error_metadata() -> None:
    assert (
        failure_case_handled(
            {
                "status": "failed",
                "error_type": "ValueError",
                "error_message": "Unsupported expression",
            }
        )
        is True
    )

    assert failure_case_handled({"status": "completed"}) is False


def test_summarize_results_separates_failure_cases_from_normal_success() -> None:
    results = [
        AgentEvalResult(
            task="Calculate 18 * 23.",
            category="calculator_basic",
            difficulty="easy",
            run_status="completed",
            task_success=True,
            tool_correct=True,
            controlled_failure=None,
            step_count=3,
            duration_ms=100,
            total_tokens=50,
        ),
        AgentEvalResult(
            task="Calculate 18 * unknown_variable.",
            category="failure_cases",
            difficulty="hard",
            run_status="failed",
            task_success=False,
            tool_correct=True,
            controlled_failure=True,
            step_count=2,
            duration_ms=50,
            total_tokens=20,
        ),
    ]

    summary = summarize_results(results)

    assert summary["runs"] == 2
    assert summary["normal_task_success"] == 1.0
    assert summary["tool_accuracy"] == 1.0
    assert summary["controlled_failure_rate"] == 1.0
