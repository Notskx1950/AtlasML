"""Abstract base class for model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Interface that every servable model must implement."""

    @abstractmethod
    async def predict(self, inputs: list[dict]) -> list[dict]:
        """Run inference on a batch of inputs."""
        ...

    @abstractmethod
    def schema_validate(self, output: dict) -> bool:
        """Check whether a single output conforms to the expected schema."""
        ...
