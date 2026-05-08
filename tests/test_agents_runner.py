"""Tests for minimal agent runner and trace persistence."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.runner import _build_tool_prompt
from app.agents.schemas import ToolCall, normalize_tool_call

# -- Fake Adapters for Testing --
class FakeToolCallingAdapter:
    """Fake LLM adapter that first returns a tool call, then a final answer."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_input_tokens = 10
        self.last_output_tokens = 20

    async def predict(self, inputs: list[dict]) -> list[dict]:
        self.calls += 1

        if self.calls == 1:
            return [
                {
                    "response": (
                        '{"tool_name": "calculator", '
                        '"arguments": {"expression": "18*23"}}'
                    )
                }
            ]

        return [{"response": "18 * 23 = 414"}]

    def schema_validate(self, output: dict) -> bool:
        return True

class FakeBadJSONAdapter:
    """Fake adapter that returns invalid tool-call JSON."""

    def __init__(self) -> None:
        self.last_input_tokens = 5
        self.last_output_tokens = 5

    async def predict(self, inputs: list[dict]) -> list[dict]:
        return [{"response": "not valid json"}]

    def schema_validate(self, output: dict) -> bool:
        return False

class FakeAliasToolCallingAdapter:
    """Fake adapter that uses common aliases for tool name and arguments."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_input_tokens = 5
        self.last_output_tokens = 5

    async def predict(self, inputs: list[dict]) -> list[dict]:
        self.calls += 1

        if self.calls == 1:
            return [
                {
                    "response": (
                        '{"tool_name": "get_model_stats", '
                        '"arguments": {"model": "llm-benchmark"}}'
                    )
                }
            ]

        return [{"response": "stats for llm-benchmark"}]

    def schema_validate(self, output: dict) -> bool:
        return True

# -- Test Cases --
def test_tool_prompt_includes_schema_and_examples() -> None:
    prompt = _build_tool_prompt("Calculate 18 * 23.")

    assert "Return ONLY valid JSON." in prompt
    assert '"tool_name": "calculator" | "echo_json" | "model_stats_lookup"' in prompt
    assert 'calculator: {"expression": "string containing only arithmetic expression"}' in prompt
    assert 'Output: {"tool_name":"calculator","arguments":{"expression":"18*23"}}' in prompt
    assert "Task: Calculate 18 * 23." in prompt


def test_normalize_tool_call_maps_tool_and_argument_aliases() -> None:
    tool_call = ToolCall(
        tool_name="get_model_stats",
        arguments={"model": "llm-benchmark"},
    )

    normalized = normalize_tool_call(tool_call)

    assert normalized.tool_name == "model_stats_lookup"
    assert normalized.arguments == {"model_name": "llm-benchmark"}


@pytest.mark.asyncio
async def test_agent_run_executes_tool_and_persists_trace(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """POST /agents/run executes a fake tool-calling flow and stores trace."""
    fake_adapter = FakeToolCallingAdapter()

    with patch(
        "app.agents.runner.RegistryStore.load",
        new=AsyncMock(return_value=fake_adapter),
    ):
        resp = await client.post(
            "/agents/run",
            json={
                "task": "Calculate 18 * 23 and explain the result.",
                "model_name": "llm-benchmark",
                "version": "v1",
            },
        )

    assert resp.status_code == 200

    run = resp.json()

    assert run["status"] == "completed"
    assert run["model_name"] == "llm-benchmark"
    assert run["model_version"] == "v1"
    assert "414" in run["final_output"]

    # Fake adapter is called twice:
    # 1. produce tool call
    # 2. produce final answer
    assert run["input_tokens"] == 20
    assert run["output_tokens"] == 40
    assert run["total_tokens"] == 60
    assert run["duration_ms"] is not None
    assert run["duration_ms"] >= 0

    run_id = run["id"]

    trace_resp = await client.get(f"/agents/runs/{run_id}/trace")
    assert trace_resp.status_code == 200

    trace = trace_resp.json()

    assert trace["run"]["id"] == run_id
    assert trace["run"]["status"] == "completed"

    steps = trace["steps"]
    step_types = [step["step_type"] for step in steps]

    assert step_types == ["llm_call", "tool_call", "final"]

    tool_invocations = trace["tool_invocations"]

    assert len(tool_invocations) == 1

    tool = tool_invocations[0]
    assert tool["tool_name"] == "calculator"
    assert tool["tool_input"] == {"expression": "18*23"}
    assert tool["tool_output"] == {"result": 414}
    assert tool["status"] == "completed"
    assert tool["latency_ms"] is not None


@pytest.mark.asyncio
async def test_agent_run_normalizes_tool_aliases_before_execution(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """AgentRunner normalizes common tool-call aliases before execution."""
    fake_adapter = FakeAliasToolCallingAdapter()

    with patch(
        "app.agents.runner.RegistryStore.load",
        new=AsyncMock(return_value=fake_adapter),
    ):
        resp = await client.post(
            "/agents/run",
            json={
                "task": "Get model statistics for llm-benchmark.",
                "model_name": "llm-benchmark",
                "version": "v1",
            },
        )

    assert resp.status_code == 200
    run = resp.json()
    assert run["status"] == "completed"

    trace_resp = await client.get(f"/agents/runs/{run['id']}/trace")
    assert trace_resp.status_code == 200
    trace = trace_resp.json()

    tool = trace["tool_invocations"][0]
    assert tool["tool_name"] == "model_stats_lookup"
    assert tool["tool_input"] == {"model_name": "llm-benchmark"}
    assert tool["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_run_records_failed_trace_on_bad_tool_json(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """AgentRunner marks run failed when tool-call JSON cannot be parsed."""
    fake_adapter = FakeBadJSONAdapter()

    with patch(
        "app.agents.runner.RegistryStore.load",
        new=AsyncMock(return_value=fake_adapter),
    ):
        resp = await client.post(
            "/agents/run",
            json={
                "task": "Calculate 18 * 23.",
                "model_name": "llm-benchmark",
                "version": "v1",
            },
        )

    assert resp.status_code == 200

    run = resp.json()

    assert run["status"] == "failed"
    assert run["error_type"] is not None
    assert run["error_message"] is not None
    assert run["duration_ms"] is not None

    trace_resp = await client.get(f"/agents/runs/{run['id']}/trace")
    assert trace_resp.status_code == 200

    trace = trace_resp.json()
    step_types = [step["step_type"] for step in trace["steps"]]

    assert "error" in step_types
