"""Tests for minimal agent runner and trace persistence."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

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

# -- Test Cases --
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