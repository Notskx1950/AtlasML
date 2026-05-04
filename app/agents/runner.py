"""Minimal tool-calling agent runner."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas import ToolCall
from app.agents.tools import execute_tool
from app.db.models import AgentRun, AgentStep, ToolInvocation
from app.models.registry_store import RegistryStore


def _duration_ms(started_at: datetime, finished_at: datetime) -> int:
    return int((finished_at - started_at).total_seconds() * 1000)


class AgentRunner:
    """Minimal single-tool-call agent runner."""

    async def run(
        self,
        task: str,
        model_name: str,
        model_version: str,
        db: AsyncSession,
    ) -> AgentRun:
        started_at = datetime.now(timezone.utc)

        run = AgentRun(
            task=task,
            model_name=model_name,
            model_version=model_version,
            status="running",
            started_at=started_at,
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)

        total_input_tokens = 0
        total_output_tokens = 0

        try:
            store = RegistryStore()
            adapter = await store.load(model_name, model_version, db)

            # Step 0: ask LLM to produce tool call JSON
            tool_prompt = (
                "You are a tool-calling agent. "
                "Return only valid JSON with keys tool_name and arguments. "
                "Available tools: calculator(expression), echo_json(payload), "
                "model_stats_lookup(model_name). "
                f"Task: {task}"
            )

            step_start = time.perf_counter()
            tool_response = await adapter.predict([{"prompt": tool_prompt}])
            step_latency = (time.perf_counter() - step_start) * 1000

            total_input_tokens += getattr(adapter, "last_input_tokens", 0)
            total_output_tokens += getattr(adapter, "last_output_tokens", 0)

            raw_tool_response = tool_response[0].get("response", "")
            tool_payload = json.loads(raw_tool_response)
            tool_call = ToolCall.model_validate(tool_payload)

            step = AgentStep(
                run_id=run.id,
                step_index=0,
                step_type="llm_call",
                input={"prompt": tool_prompt},
                output={"raw_response": raw_tool_response},
                status="completed",
                latency_ms=step_latency,
            )
            db.add(step)
            await db.commit()
            await db.refresh(step)

            # Step 1: execute tool
            tool_start = time.perf_counter()
            try:
                tool_output = await execute_tool(
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    db=db,
                )
                tool_status = "completed"
                tool_error = None
            except Exception as exc:
                tool_output = None
                tool_status = "failed"
                tool_error = str(exc)

            tool_latency = (time.perf_counter() - tool_start) * 1000

            invocation = ToolInvocation(
                run_id=run.id,
                step_id=step.id,
                tool_name=tool_call.tool_name,
                tool_input=tool_call.arguments,
                tool_output=tool_output,
                status=tool_status,
                error_message=tool_error,
                latency_ms=tool_latency,
            )
            db.add(invocation)

            tool_step = AgentStep(
                run_id=run.id,
                step_index=1,
                step_type="tool_call",
                input={
                    "tool_name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                },
                output=tool_output if tool_output is not None else {"error": tool_error},
                status=tool_status,
                latency_ms=tool_latency,
            )
            db.add(tool_step)
            await db.commit()

            if tool_status == "failed":
                raise ValueError(tool_error or "Tool execution failed")

            # Step 2: final answer
            final_prompt = (
                "Use the tool result to answer the task. "
                f"Task: {task}\n"
                f"Tool result: {json.dumps(tool_output)}"
            )

            final_start = time.perf_counter()
            final_response = await adapter.predict([{"prompt": final_prompt}])
            final_latency = (time.perf_counter() - final_start) * 1000

            total_input_tokens += getattr(adapter, "last_input_tokens", 0)
            total_output_tokens += getattr(adapter, "last_output_tokens", 0)

            final_output = final_response[0].get("response", "")

            final_step = AgentStep(
                run_id=run.id,
                step_index=2,
                step_type="final",
                input={"prompt": final_prompt},
                output={"response": final_output},
                status="completed",
                latency_ms=final_latency,
            )
            db.add(final_step)

            finished_at = datetime.now(timezone.utc)
            run.status = "completed"
            run.final_output = final_output
            run.finished_at = finished_at
            run.duration_ms = _duration_ms(started_at, finished_at)
            run.input_tokens = total_input_tokens
            run.output_tokens = total_output_tokens
            run.total_tokens = total_input_tokens + total_output_tokens

            await db.commit()
            await db.refresh(run)
            return run

        except Exception as exc:
            finished_at = datetime.now(timezone.utc)
            run.status = "failed"
            run.error_type = type(exc).__name__
            run.error_message = str(exc)
            run.finished_at = finished_at
            run.duration_ms = _duration_ms(started_at, finished_at)
            run.input_tokens = total_input_tokens
            run.output_tokens = total_output_tokens
            run.total_tokens = total_input_tokens + total_output_tokens

            db.add(
                AgentStep(
                    run_id=run.id,
                    step_index=999,
                    step_type="error",
                    input={"task": task},
                    output={"error": str(exc)},
                    status="failed",
                )
            )

            await db.commit()
            await db.refresh(run)
            return run