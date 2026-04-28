"""
Agent-style tool-calling adapter for AtlasML.

This adapter implements the same ModelAdapter interface as other servable
models in AtlasML, but instead of running a classical ML model, it executes
a lightweight tool-calling workflow.

The goal is to demonstrate how AtlasML's model-serving abstraction can be
extended toward agent-style execution, structured tool calls, and task-level
tracing.
"""

from __future__ import annotations

import ast
import operator
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from app.models.base import ModelAdapter


class ToolCall(BaseModel):
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    step_index: int
    step_type: str
    content: dict[str, Any]
    status: str
    latency_ms: float = 0.0
    error_type: str | None = None
    error_message: str | None = None


class AgentRunOutput(BaseModel):
    task: str
    final_answer: str | None = None
    status: str
    steps: list[AgentStep] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    total_latency_ms: float = 0.0


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def execute(self, tool_call: ToolCall) -> Any:
        if tool_call.tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_call.tool_name}")

        tool = self.tools[tool_call.tool_name]
        return tool.func(**tool_call.tool_args)


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _safe_eval_expr(node.operand)
        return _ALLOWED_OPERATORS[type(node.op)](operand)

    raise ValueError("Unsupported or unsafe expression")


def calculator(expression: str) -> float:
    parsed = ast.parse(expression, mode="eval")
    return _safe_eval_expr(parsed.body)


def string_length(text: str) -> int:
    return len(text)


def lookup_model_status(model_name: str) -> dict[str, Any]:
    mock_registry = {
        "demo_classifier": {
            "active_version": "v2",
            "framework": "sklearn",
            "task_type": "classification",
            "status": "active",
        },
        "llm_json_extractor": {
            "active_version": "v1",
            "framework": "openai-compatible",
            "task_type": "structured_extraction",
            "status": "active",
        },
    }

    if model_name not in mock_registry:
        raise ValueError(f"Model not found: {model_name}")

    return mock_registry[model_name]


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="calculator",
            description="Safely evaluate a basic arithmetic expression.",
            func=calculator,
        )
    )
    registry.register(
        Tool(
            name="string_length",
            description="Return the length of a string.",
            func=string_length,
        )
    )
    registry.register(
        Tool(
            name="lookup_model_status",
            description="Look up mock AtlasML model registry status.",
            func=lookup_model_status,
        )
    )

    return registry


class AgentToolAdapter(ModelAdapter):
    """
    AtlasML-compatible adapter for lightweight agent-style tool calling.

    Input format:
    {
        "task": "Calculate a simple expression"
    }

    Output format:
    {
        "task": "...",
        "final_answer": "...",
        "status": "success",
        "steps": [...],
        "metrics": {...},
        "total_latency_ms": ...
    }
    """

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        self.tool_registry = tool_registry or build_default_tool_registry()

    async def predict(self, inputs: list[dict]) -> list[dict]:
        outputs = []

        for item in inputs:
            task = item.get("task")
            if not task:
                outputs.append(
                    AgentRunOutput(
                        task="",
                        final_answer=None,
                        status="failed",
                        metrics={"task_success": False},
                        steps=[
                            AgentStep(
                                step_index=1,
                                step_type="input_validation",
                                content=item,
                                status="failed",
                                error_type="missing_task",
                                error_message="Input must include a `task` field.",
                            )
                        ],
                    ).model_dump()
                )
                continue

            outputs.append(self._run_task(task).model_dump())

        return outputs

    def schema_validate(self, output: dict) -> bool:
        try:
            AgentRunOutput.model_validate(output)
            return True
        except ValidationError:
            return False

    def _run_task(self, task: str) -> AgentRunOutput:
        run_start = time.perf_counter()
        steps: list[AgentStep] = []

        try:
            plan_start = time.perf_counter()
            tool_call = self._plan(task)
            steps.append(
                AgentStep(
                    step_index=1,
                    step_type="plan",
                    content={"tool_call": tool_call.model_dump()},
                    status="success",
                    latency_ms=(time.perf_counter() - plan_start) * 1000,
                )
            )

            tool_start = time.perf_counter()
            tool_result = self.tool_registry.execute(tool_call)
            steps.append(
                AgentStep(
                    step_index=2,
                    step_type="tool_call",
                    content={
                        "tool_name": tool_call.tool_name,
                        "tool_args": tool_call.tool_args,
                    },
                    status="success",
                    latency_ms=(time.perf_counter() - tool_start) * 1000,
                )
            )

            steps.append(
                AgentStep(
                    step_index=3,
                    step_type="tool_result",
                    content={"result": tool_result},
                    status="success",
                )
            )

            final_answer = f"Tool `{tool_call.tool_name}` returned: {tool_result}"
            steps.append(
                AgentStep(
                    step_index=4,
                    step_type="final",
                    content={"answer": final_answer},
                    status="success",
                )
            )

            total_latency_ms = (time.perf_counter() - run_start) * 1000
            metrics = self._compute_metrics(
                status="success",
                steps=steps,
                total_latency_ms=total_latency_ms,
            )

            return AgentRunOutput(
                task=task,
                final_answer=final_answer,
                status="success",
                steps=steps,
                metrics=metrics,
                total_latency_ms=total_latency_ms,
            )

        except Exception as exc:
            steps.append(
                AgentStep(
                    step_index=len(steps) + 1,
                    step_type="final",
                    content={},
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

            total_latency_ms = (time.perf_counter() - run_start) * 1000
            metrics = self._compute_metrics(
                status="failed",
                steps=steps,
                total_latency_ms=total_latency_ms,
            )

            return AgentRunOutput(
                task=task,
                final_answer=None,
                status="failed",
                steps=steps,
                metrics=metrics,
                total_latency_ms=total_latency_ms,
            )

    def _plan(self, task: str) -> ToolCall:
        normalized = task.lower()

        if "calculate" in normalized or "compute" in normalized:
            return ToolCall(
                tool_name="calculator",
                tool_args={"expression": "12 * 7 + 3"},
            )

        if "length" in normalized:
            return ToolCall(
                tool_name="string_length",
                tool_args={"text": "AtlasML agent demo"},
            )

        if "model" in normalized or "registry" in normalized:
            return ToolCall(
                tool_name="lookup_model_status",
                tool_args={"model_name": "demo_classifier"},
            )

        raise ValueError("Planner could not select a tool for this task")

    def _compute_metrics(
        self,
        status: str,
        steps: list[AgentStep],
        total_latency_ms: float,
    ) -> dict[str, Any]:
        tool_steps = [s for s in steps if s.step_type == "tool_call"]
        failed_steps = [s for s in steps if s.status == "failed"]

        return {
            "task_success": status == "success",
            "step_count": len(steps),
            "tool_call_count": len(tool_steps),
            "failed_step_count": len(failed_steps),
            "latency_ms": round(total_latency_ms, 3),
        }