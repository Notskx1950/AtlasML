"""Shared test fixtures for AtlasML."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.models.base import ModelAdapter

# Use SQLite for tests (in-memory)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a session-scoped event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_engine():
    """Create an async test engine with all tables."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional test DB session."""
    session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(async_engine, db_session) -> AsyncGenerator[AsyncClient, None]:
    """Provide an httpx async client wired to the test app."""
    from app.api.deps import get_db
    from app.main import create_app

    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class MockSklearnAdapter(ModelAdapter):
    """A mock sklearn adapter for testing."""

    async def predict(self, inputs: list[dict]) -> list[dict]:
        """Return a fixed prediction for each input."""
        return [{"prediction": 1} for _ in inputs]

    def schema_validate(self, output: dict) -> bool:
        """Always valid."""
        return "prediction" in output


class BrokenAdapter(ModelAdapter):
    """An adapter that always fails."""

    async def predict(self, inputs: list[dict]) -> list[dict]:
        """Raise an error."""
        raise FileNotFoundError("Model artifact not found: /bad/path.joblib")

    def schema_validate(self, output: dict) -> bool:
        """Always invalid."""
        return False


class SchemaFailAdapter(ModelAdapter):
    """An adapter whose output fails schema validation."""

    async def predict(self, inputs: list[dict]) -> list[dict]:
        """Return output missing the expected key."""
        return [{"wrong_key": "bad"} for _ in inputs]

    def schema_validate(self, output: dict) -> bool:
        """Always fails validation."""
        return False


@pytest.fixture
def mock_adapter() -> MockSklearnAdapter:
    """Return a mock sklearn adapter."""
    return MockSklearnAdapter()


@pytest.fixture
def broken_adapter() -> BrokenAdapter:
    """Return a broken adapter."""
    return BrokenAdapter()


@pytest.fixture
def schema_fail_adapter() -> SchemaFailAdapter:
    """Return an adapter with schema validation failure."""
    return SchemaFailAdapter()


@pytest.fixture
def sample_dataset() -> Generator[str, None, None]:
    """Create a temporary NDJSON dataset file."""
    rows = [
        {"input": {"feature_a": i, "feature_b": i * 2}, "label": i % 2}
        for i in range(10)
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        path = f.name
    yield path
    os.unlink(path)
