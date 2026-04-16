"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session and ensure cleanup."""
    async with get_async_session_factory()() as session:
        yield session
