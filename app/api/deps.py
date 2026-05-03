"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session and ensure cleanup."""
    async with get_async_session_factory()() as session:
        yield session

def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require X-API-Key when API_KEY is configured.

    If settings.API_KEY is unset, auth is disabled for local development/tests.
    """
    if not settings.API_KEY:
        return

    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )