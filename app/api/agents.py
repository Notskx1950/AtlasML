"""Agent execution and trace API routes."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.models import AgentRun, AgentStep, ToolInvocation, ModelVersion
from app.agents.runner import AgentRunner

router = APIRouter(prefix="/agents")

# --- Schemas ---
class AgentRunRequest(BaseModel):
    task: str
    model_name: str
    version: str | None = None

class AgentStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    step_index: int
    step_type: str
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    status: str
    latency_ms: float | None = None
    created_at: datetime


class ToolInvocationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    step_id: uuid.UUID | None = None
    tool_name: str
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None
    status: str
    error_message: str | None = None
    latency_ms: float | None = None
    created_at: datetime


class AgentRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    task: str
    model_name: str
    model_version: str
    status: str
    final_output: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None


class AgentTraceResponse(BaseModel):
    run: AgentRunResponse
    steps: list[AgentStepResponse] = Field(default_factory=list)
    tool_invocations: list[ToolInvocationResponse] = Field(default_factory=list)

class AgentStatsResponse(BaseModel):
    total_runs: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_steps: float | None = None
    avg_duration_ms: float | None = None
    avg_total_tokens: float | None = None
    top_tools: dict[str, int]
    top_error_types: dict[str, int]
# --- Routes ---
@router.get("/runs/{run_id}", response_model=AgentRunResponse)
async def get_agent_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> AgentRun:
    run = await db.get(AgentRun, uuid.UUID(run_id))
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run {run_id} not found",
        )
    return run


@router.get("/runs/{run_id}/trace", response_model=AgentTraceResponse)
async def get_agent_trace(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> AgentTraceResponse:
    run = await db.get(AgentRun, uuid.UUID(run_id))
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run {run_id} not found",
        )

    steps_result = await db.execute(
        select(AgentStep)
        .where(AgentStep.run_id == run.id)
        .order_by(AgentStep.step_index.asc())
    )
    steps = list(steps_result.scalars().all())

    tools_result = await db.execute(
        select(ToolInvocation)
        .where(ToolInvocation.run_id == run.id)
        .order_by(ToolInvocation.created_at.asc())
    )
    tool_invocations = list(tools_result.scalars().all())

    return AgentTraceResponse(
        run=run,
        steps=steps,
        tool_invocations=tool_invocations,
    )

@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    body: AgentRunRequest,
    db: AsyncSession = Depends(get_db),
) -> AgentRun:
    version = body.version

    if version is None:
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.name == body.model_name,
                ModelVersion.is_active == True,  # noqa: E712
            )
        )
        active = result.scalar_one_or_none()
        if active is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active version for model {body.model_name}",
            )
        version = active.version

    runner = AgentRunner()
    return await runner.run(
        task=body.task,
        model_name=body.model_name,
        model_version=version,
        db=db,
    )

@router.get("/stats", response_model=AgentStatsResponse)
async def get_agent_stats(
    db: AsyncSession = Depends(get_db),
) -> AgentStatsResponse:
    runs_result = await db.execute(select(AgentRun))
    runs = list(runs_result.scalars().all())

    steps_result = await db.execute(select(AgentStep))
    steps = list(steps_result.scalars().all())

    tools_result = await db.execute(select(ToolInvocation))
    tools = list(tools_result.scalars().all())

    total_runs = len(runs)
    success_count = sum(1 for run in runs if run.status == "completed")
    failure_count = sum(1 for run in runs if run.status == "failed")
    success_rate = success_count / total_runs if total_runs else 0.0

    durations = [run.duration_ms for run in runs if run.duration_ms is not None]
    tokens = [run.total_tokens for run in runs if run.total_tokens is not None]

    steps_by_run: dict[uuid.UUID, int] = {}
    for step in steps:
        steps_by_run[step.run_id] = steps_by_run.get(step.run_id, 0) + 1

    avg_steps = (
        sum(steps_by_run.values()) / len(steps_by_run)
        if steps_by_run
        else None
    )

    tool_counter = Counter(tool.tool_name for tool in tools)

    error_counter = Counter(
        run.error_type for run in runs if run.error_type is not None
    )

    return AgentStatsResponse(
        total_runs=total_runs,
        success_count=success_count,
        failure_count=failure_count,
        success_rate=round(success_rate, 4),
        avg_steps=round(avg_steps, 2) if avg_steps is not None else None,
        avg_duration_ms=round(sum(durations) / len(durations), 2)
        if durations
        else None,
        avg_total_tokens=round(sum(tokens) / len(tokens), 2)
        if tokens
        else None,
        top_tools=dict(tool_counter),
        top_error_types=dict(error_counter),
    )