"""Inference API routes — sync and async predict."""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from redis import Redis
from rq import Queue

from app.api.deps import get_db
from app.config import settings
from app.db.models import InferenceLog, JobRecord, ModelVersion
from app.models.registry_store import RegistryStore

logger = structlog.get_logger()
router = APIRouter()


# --- Pydantic schemas ---


class PredictRequest(BaseModel):
    """Request body for /predict."""

    model_name: str
    inputs: list[dict[str, Any]]
    version: str | None = None


class PredictResponse(BaseModel):
    """Response body for sync predict."""

    predictions: list[dict[str, Any]]
    model_version: str
    latency_ms: float


class JobCreatedResponse(BaseModel):
    """Response body for async job creation."""

    job_id: str


class JobStatusResponse(BaseModel):
    """Response body for job status queries."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    job_type: str
    model_name: str
    model_version: str
    status: str
    error_msg: str | None = None
    result: dict | None = None


# --- Helpers ---


async def _resolve_version(
    model_name: str, version: str | None, db: AsyncSession
) -> str:
    """Resolve the model version, defaulting to the active version."""
    if version:
        return version
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == model_name, ModelVersion.is_active == True  # noqa: E712
        )
    )
    mv = result.scalar_one_or_none()
    if mv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active version for model {model_name}",
        )
    return mv.version


# --- Routes ---


@router.post("/predict", response_model=PredictResponse)
async def sync_predict(
    body: PredictRequest, db: AsyncSession = Depends(get_db)
) -> PredictResponse:
    """Run synchronous inference and log the result."""
    version = await _resolve_version(body.model_name, body.version, db)
    store = RegistryStore()

    start = time.perf_counter()
    status_code = 200
    error_type: str | None = None
    schema_valid: bool | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    predictions: list[dict[str, Any]] = []

    try:
        adapter = await store.load(body.model_name, version, db)
        predictions = await adapter.predict(body.inputs)

        # Check schema validity on the first output
        if predictions:
            schema_valid = adapter.schema_validate(predictions[0])

        # Capture token counts if available (LLM adapter)
        if hasattr(adapter, "last_input_tokens"):
            input_tokens = adapter.last_input_tokens
            output_tokens = adapter.last_output_tokens

    except FileNotFoundError as exc:
        status_code = 422
        error_type = "FileNotFoundError"
        logger.error("predict_failed", error=str(exc), model=body.model_name)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        status_code = 404
        error_type = "ValueError"
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        status_code = 500
        error_type = type(exc).__name__
        logger.error("predict_failed", error=str(exc), model=body.model_name)
        raise HTTPException(status_code=500, detail="Internal inference error") from exc
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        log = InferenceLog(
            model_name=body.model_name,
            model_version=version,
            latency_ms=latency_ms,
            status_code=status_code,
            error_type=error_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            schema_valid=schema_valid,
        )
        db.add(log)
        await db.commit()

    logger.info(
        "prediction_complete",
        model=body.model_name,
        version=version,
        latency_ms=round(latency_ms, 2),
        schema_valid=schema_valid,
    )

    return PredictResponse(
        predictions=predictions,
        model_version=version,
        latency_ms=round(latency_ms, 2),
    )


@router.post("/jobs/predict", status_code=status.HTTP_202_ACCEPTED, response_model=JobCreatedResponse)
async def async_predict(
    body: PredictRequest, db: AsyncSession = Depends(get_db)
) -> JobCreatedResponse:
    """Enqueue an async prediction job via RQ."""
    version = await _resolve_version(body.model_name, body.version, db)
    job_id = str(uuid.uuid4())

    # Create job record
    record = JobRecord(
        id=uuid.UUID(job_id),
        job_type="predict",
        model_name=body.model_name,
        model_version=version,
        status="queued",
    )
    db.add(record)
    await db.commit()

    # Enqueue RQ job
    redis_conn = Redis.from_url(settings.REDIS_URL)
    q = Queue(connection=redis_conn)
    q.enqueue(
        "app.workers.tasks.run_predict",
        job_id=job_id,
        model_name=body.model_name,
        version=version,
        inputs=body.inputs,
    )

    return JobCreatedResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str, db: AsyncSession = Depends(get_db)
) -> JobRecord:
    """Look up the status of an async job."""
    result = await db.execute(
        select(JobRecord).where(JobRecord.id == uuid.UUID(job_id))
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return record
