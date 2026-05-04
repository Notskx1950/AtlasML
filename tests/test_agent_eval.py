"""Tests for agent eval scoring helpers."""

from __future__ import annotations

from scripts.run_agent_eval import contains_expected_answer, tool_called_correctly


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