"""Async and sync database engine and session factories."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


@lru_cache(maxsize=1)
def _get_async_engine():
    """Lazily create the async engine."""
    return create_async_engine(settings.DATABASE_URL, echo=False, future=True)


@lru_cache(maxsize=1)
def _get_async_session_factory():
    """Lazily create the async session factory."""
    return async_sessionmaker(
        _get_async_engine(), class_=AsyncSession, expire_on_commit=False
    )


@lru_cache(maxsize=1)
def _get_sync_engine():
    """Lazily create the sync engine."""
    return create_engine(settings.DATABASE_SYNC_URL, echo=False, future=True)


@lru_cache(maxsize=1)
def _get_sync_session_factory():
    """Lazily create the sync session factory."""
    return sessionmaker(_get_sync_engine(), class_=Session, expire_on_commit=False)


def get_async_session_factory():
    """Return the async session factory."""
    return _get_async_session_factory()


def get_sync_session_factory():
    """Return the sync session factory."""
    return _get_sync_session_factory()

