"""Schemas for agent execution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


class AgentRunnerResult(BaseModel):
    run_id: str
    status: str
    final_output: str | None = None