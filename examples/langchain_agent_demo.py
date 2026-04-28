"""
Run a lightweight LangChain-style tool-calling demo using AtlasML's
ModelAdapter interface.

This example demonstrates an agent-style adapter that can be used like
other AtlasML model adapters: it receives batched inputs and returns
validated structured outputs.
"""

from __future__ import annotations

import asyncio
import json

from app.models.agent_tool_adapter import AgentToolAdapter


async def main() -> None:
    adapter = AgentToolAdapter()

    inputs = [
        {"task": "Calculate a simple expression."},
        {"task": "What is the length of this phrase?"},
        {"task": "Check the active model in the registry."},
        {"task": "Do something unsupported."},
        {},
    ]

    outputs = await adapter.predict(inputs)

    for idx, output in enumerate(outputs, start=1):
        print("=" * 80)
        print(f"Example {idx}")
        print(json.dumps(output, indent=2))

        is_valid = adapter.schema_validate(output)
        print(f"schema_valid: {is_valid}")


if __name__ == "__main__":
    asyncio.run(main())