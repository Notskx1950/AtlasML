"""FastAPI application factory."""

from __future__ import annotations

import logging
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="AtlasML", version="0.1.0")

    # --- Structured logging ---
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # --- Request ID middleware ---
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Attach a unique request_id to every request context."""
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # --- Prometheus ---
    Instrumentator().instrument(app).expose(app)

    # --- Register routers ---
    from app.api.registry import router as registry_router
    from app.api.inference import router as inference_router
    from app.api.eval import router as eval_router

    app.include_router(registry_router, tags=["registry"])
    app.include_router(inference_router, tags=["inference"])
    app.include_router(eval_router, tags=["eval"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()
