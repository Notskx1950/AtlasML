"""Schemas for agent execution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

TOOL_ALIASES = {
    "get_model_stats": "model_stats_lookup",
    "lookup_model_stats": "model_stats_lookup",
    "model_stats": "model_stats_lookup",
}

ARG_ALIASES = {
    "model": "model_name",
    "name": "model_name",
}


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


def normalize_tool_call(tool_call: ToolCall) -> ToolCall:
    """Normalize common LLM tool-call aliases before execution."""
    tool_name = tool_call.tool_name.strip().lower()
    normalized_tool_name = TOOL_ALIASES.get(tool_name, tool_name)
    arguments = dict(tool_call.arguments)

    if normalized_tool_name == "model_stats_lookup":
        for alias, canonical in ARG_ALIASES.items():
            if canonical not in arguments and alias in arguments:
                arguments[canonical] = arguments.pop(alias)

    return ToolCall(tool_name=normalized_tool_name, arguments=arguments)


class AgentRunnerResult(BaseModel):
    run_id: str
    status: str
    final_output: str | None = None
