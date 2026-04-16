"""Adapter for serving scikit-learn models serialized with joblib."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import joblib

from app.models.base import ModelAdapter


class SklearnAdapter(ModelAdapter):
    """Loads and serves a joblib-serialized sklearn model."""

    def __init__(self, artifact_uri: str, runtime_config: dict[str, Any] | None = None) -> None:
        """Initialize with path to the serialized model."""
        self._artifact_uri = artifact_uri
        self._runtime_config = runtime_config or {}
        self._model: Any = None

    def _load(self) -> None:
        """Load the model from disk."""
        path = Path(self._artifact_uri)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self._artifact_uri}")
        self._model = joblib.load(path)

    @property
    def model(self) -> Any:
        """Return the loaded sklearn model, loading lazily if needed."""
        if self._model is None:
            self._load()
        return self._model

    async def predict(self, inputs: list[dict]) -> list[dict]:
        """Run sklearn predict on a list of feature dicts."""
        import pandas as pd

        df = pd.DataFrame(inputs)
        loop = asyncio.get_running_loop()
        predictions = await loop.run_in_executor(None, self.model.predict, df)
        return [{"prediction": p} for p in predictions.tolist()]

    def schema_validate(self, output: dict) -> bool:
        """Validate that the output contains a prediction key."""
        return "prediction" in output
