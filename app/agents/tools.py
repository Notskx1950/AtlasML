"""Small built-in tools for AtlasML agents."""

from __future__ import annotations

import ast
import operator
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import InferenceLog, ModelVersion


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _eval_expr(node.left)
        right = _eval_expr(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    raise ValueError("Unsupported expression")


def calculator(expression: str) -> dict[str, Any]:
    """Safely evaluate a simple arithmetic expression."""
    tree = ast.parse(expression, mode="eval")
    return {"result": _eval_expr(tree)}


def echo_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Echo structured JSON payload."""
    return payload


async def model_stats_lookup(
    model_name: str,
    db: AsyncSession,
) -> dict[str, Any]:
    """Return a lightweight model stats summary."""
    active_result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == model_name,
            ModelVersion.is_active == True,  # noqa: E712
        )
    )
    active = active_result.scalar_one_or_none()

    logs_result = await db.execute(
        select(InferenceLog).where(InferenceLog.model_name == model_name)
    )
    logs = list(logs_result.scalars().all())

    total = len(logs)
    error_count = sum(1 for log in logs if log.status_code >= 400)

    return {
        "model_name": model_name,
        "active_version": active.version if active else None,
        "total_predictions": total,
        "error_count": error_count,
    }


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    db: AsyncSession,
) -> dict[str, Any]:
    if tool_name == "calculator":
        return calculator(**arguments)

    if tool_name == "echo_json":
        return echo_json(**arguments)

    if tool_name == "model_stats_lookup":
        return await model_stats_lookup(db=db, **arguments)

    raise ValueError(f"Unknown tool: {tool_name}")