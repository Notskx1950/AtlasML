"""Adapter for OpenAI-compatible LLM chat completions."""

from __future__ import annotations

import json
import time
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.models.base import ModelAdapter


class LLMAdapter(ModelAdapter):
    """Calls an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        artifact_uri: str,
        runtime_config: dict[str, Any] | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> None:
        """Initialize with model name and optional output schema."""
        self._model_id = artifact_uri  # e.g. "gpt-4o-mini"
        self._runtime_config = runtime_config or {}
        self._output_schema = output_schema
        self._client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY or "sk-placeholder",
            base_url=settings.OPENAI_API_BASE,
        )
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    async def predict(self, inputs: list[dict]) -> list[dict]:
        """Send each input as a chat completion request."""
        results: list[dict] = []
        total_input_tokens = 0
        total_output_tokens = 0

        max_tokens = self._runtime_config.get("max_tokens", 1024)
        temperature = self._runtime_config.get("temperature", 0.0)

        for inp in inputs:
            prompt = inp.get("prompt", inp.get("text", json.dumps(inp)))
            messages = [{"role": "user", "content": prompt}]

            response = await self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            choice = response.choices[0]
            content = choice.message.content or ""
            usage = response.usage

            if usage:
                total_input_tokens += usage.prompt_tokens
                total_output_tokens += usage.completion_tokens

            result: dict[str, Any] = {"response": content}
            if self._output_schema is not None:
                result["schema_valid"] = self.schema_validate(result)
            results.append(result)

        self.last_input_tokens = total_input_tokens
        self.last_output_tokens = total_output_tokens
        return results

    def schema_validate(self, output: dict) -> bool:
        """Validate the response against the configured Pydantic schema."""
        if self._output_schema is None:
            return True
        try:
            raw = output.get("response", "")
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            self._output_schema.model_validate(parsed)
            return True
        except (json.JSONDecodeError, ValidationError):
            return False
