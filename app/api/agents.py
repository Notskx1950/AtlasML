"""Agent execution and trace API routes."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

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