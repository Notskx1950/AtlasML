"""In-process model cache mapping (name, version) to a loaded ModelAdapter."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ModelVersion
from app.models.base import ModelAdapter
from app.models.sklearn_adapter import SklearnAdapter
from app.models.llm_adapter import LLMAdapter


class RegistryStore:
    """Singleton cache of loaded model adapters."""

    _instance: RegistryStore | None = None
    _cache: dict[tuple[str, str], ModelAdapter]

    def __new__(cls) -> RegistryStore:
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    def get(self, name: str, version: str) -> ModelAdapter | None:
        """Return a cached adapter or None."""
        return self._cache.get((name, version))

    def put(self, name: str, version: str, adapter: ModelAdapter) -> None:
        """Store an adapter in the cache."""
        self._cache[(name, version)] = adapter

    def evict(self, name: str, version: str) -> None:
        """Remove an adapter from the cache."""
        self._cache.pop((name, version), None)

    async def load(
        self, name: str, version: str, session: AsyncSession
    ) -> ModelAdapter:
        """Load an adapter from DB metadata, caching it for reuse."""
        cached = self.get(name, version)
        if cached is not None:
            return cached

        stmt = select(ModelVersion).where(
            ModelVersion.name == name, ModelVersion.version == version
        )
        result = await session.execute(stmt)
        mv = result.scalar_one_or_none()
        if mv is None:
            raise ValueError(f"Model {name}:{version} not found in registry")

        adapter = _build_adapter(mv.artifact_uri, mv.runtime_config)
        self.put(name, version, adapter)
        return adapter

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()


def _build_adapter(
    artifact_uri: str, runtime_config: dict[str, Any] | None
) -> ModelAdapter:
    """Construct the appropriate adapter based on the artifact URI."""
    cfg = runtime_config or {}
    adapter_type = cfg.get("adapter_type", "sklearn")

    if adapter_type == "llm":
        return LLMAdapter(artifact_uri=artifact_uri, runtime_config=cfg)
    return SklearnAdapter(artifact_uri=artifact_uri, runtime_config=cfg)
