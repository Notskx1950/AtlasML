from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentRun, AgentStep, ToolInvocation


@pytest.mark.asyncio
async def test_agent_stats_aggregates_runs_steps_and_tools(
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
        final_output="414",
        total_tokens=100,
        finished_at=datetime.now(timezone.utc),
        duration_ms=1200,
    )
    db_session.add(run)

    db_session.add(
        AgentStep(
            run_id=run_id,
            step_index=0,
            step_type="llm_call",
            status="completed",
        )
    )
    db_session.add(
        AgentStep(
            run_id=run_id,
            step_index=1,
            step_type="tool_call",
            status="completed",
        )
    )

    db_session.add(
        ToolInvocation(
            run_id=run_id,
            tool_name="calculator",
            tool_input={"expression": "18*23"},
            tool_output={"result": 414},
            status="completed",
        )
    )

    await db_session.commit()

    resp = await client.get("/agents/stats")

    assert resp.status_code == 200

    data = resp.json()

    assert data["total_runs"] == 1
    assert data["success_count"] == 1
    assert data["failure_count"] == 0
    assert data["success_rate"] == 1.0
    assert data["avg_steps"] == 2.0
    assert data["avg_duration_ms"] == 1200.0
    assert data["avg_total_tokens"] == 100.0
    assert data["top_tools"] == {"calculator": 1}