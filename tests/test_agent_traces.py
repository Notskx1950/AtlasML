from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentRun, AgentStep, ToolInvocation


@pytest.mark.asyncio
async def test_get_agent_trace_returns_steps_and_tools(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    run_id = uuid.uuid4()

    run = AgentRun(
        id=run_id,
        task="Calculate 18 * 23.",
        model_name="llm-benchmark",
        model_version="v1",
        status="completed",
        final_output="18 * 23 = 414",
        input_tokens=20,
        output_tokens=30,
        total_tokens=50,
        finished_at=datetime.now(timezone.utc),
        duration_ms=1234,
    )
    db_session.add(run)

    step_id = uuid.uuid4()
    db_session.add(
        AgentStep(
            id=step_id,
            run_id=run_id,
            step_index=0,
            step_type="tool_call",
            input={"task": "Calculate 18 * 23."},
            output={"tool_name": "calculator", "arguments": {"expression": "18*23"}},
            status="completed",
            latency_ms=100.0,
        )
    )

    db_session.add(
        ToolInvocation(
            run_id=run_id,
            step_id=step_id,
            tool_name="calculator",
            tool_input={"expression": "18*23"},
            tool_output={"result": 414},
            status="completed",
            latency_ms=10.0,
        )
    )

    await db_session.commit()

    resp = await client.get(f"/agents/runs/{run_id}/trace")

    assert resp.status_code == 200
    body = resp.json()

    assert body["run"]["id"] == str(run_id)
    assert body["run"]["status"] == "completed"
    assert len(body["steps"]) == 1
    assert body["steps"][0]["step_type"] == "tool_call"
    assert len(body["tool_invocations"]) == 1
    assert body["tool_invocations"][0]["tool_name"] == "calculator"


@pytest.mark.asyncio
async def test_get_missing_agent_run_returns_404(
    client: AsyncClient,
) -> None:
    missing_id = uuid.uuid4()

    resp = await client.get(f"/agents/runs/{missing_id}")

    assert resp.status_code == 404