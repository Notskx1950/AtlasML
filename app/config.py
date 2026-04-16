"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the AtlasML platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    DATABASE_URL: str = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlasml"
    DATABASE_SYNC_URL: str = "postgresql+psycopg2://atlas:atlas@localhost:5432/atlasml"
    REDIS_URL: str = "redis://localhost:6379"
    LOG_LEVEL: str = "INFO"
    DEFAULT_MODEL_TIMEOUT_MS: int = 5000
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str = ""


settings = Settings()
