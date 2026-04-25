"""Evaluation API routes."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from redis import Redis
from rq import Queue

from app.api.deps import get_db
from app.config import settings
from app.db.models import EvalMetric, EvalRun, ModelVersion

router = APIRouter(prefix="/eval")


# --- Pydantic schemas ---


class EvalRunRequest(BaseModel):
    """Request body for launching an eval run."""

    model_name: str
    version: str
    dataset_id: str
    dataset_path: str


class EvalRunResponse(BaseModel):
    """Response for eval run creation."""

    run_id: str


class EvalMetricResponse(BaseModel):
    """Single metric in eval results."""

    model_config = ConfigDict(from_attributes=True)

    metric_name: str
    value: float


class EvalRunDetailResponse(BaseModel):
    """Full eval run detail with metrics."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    model_name: str
    model_version: str
    dataset_id: str
    dataset_hash: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    metrics: list[EvalMetricResponse] = []


class ComparisonItem(BaseModel):
    """Single metric comparison between two versions."""

    metric: str
    v1: float
    v2: float
    delta: float
    pct_change: float


class ComparisonResponse(BaseModel):
    """Response for version comparison."""

    model_name: str
    v1: str
    v2: str
    comparison: list[ComparisonItem]


# --- Routes ---


@router.post("/run", status_code=status.HTTP_202_ACCEPTED, response_model=EvalRunResponse)
async def start_eval_run(
    body: EvalRunRequest, db: AsyncSession = Depends(get_db)
) -> EvalRunResponse:
    """Launch an eval run via RQ."""
    # Validate model version exists
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == body.model_name,
            ModelVersion.version == body.version,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {body.model_name}:{body.version} not found",
        )

    run_id = uuid.uuid4()
    # Generate dataset hash for easier comparison of runs using the same dataset
    dataset_hash = hashlib.sha256(body.dataset_path.encode()).hexdigest()[:16]
    
    run = EvalRun(
        id=run_id,
        model_name=body.model_name,
        model_version=body.version,
        dataset_id=body.dataset_id,
        dataset_hash=dataset_hash,
        status="running",
    )
    db.add(run)
    await db.commit()

    # Enqueue RQ job
    redis_conn = Redis.from_url(settings.REDIS_URL)
    q = Queue(connection=redis_conn)
    q.enqueue(
        "app.workers.tasks.run_eval",
        run_id=str(run_id),
        model_name=body.model_name,
        version=body.version,
        dataset_path=body.dataset_path,
    )

    return EvalRunResponse(run_id=str(run_id))


@router.get("/runs/{run_id}", response_model=EvalRunDetailResponse)
async def get_eval_run(
    run_id: str, db: AsyncSession = Depends(get_db)
) -> EvalRun:
    """Return eval run with its metrics."""
    result = await db.execute(
        select(EvalRun).where(EvalRun.id == uuid.UUID(run_id))
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Eval run {run_id} not found",
        )
    # Eagerly load metrics
    await db.refresh(run, ["metrics"])
    return run


@router.get("/compare", response_model=ComparisonResponse)
async def compare_versions(
    model_name: str = Query(...),
    v1: str = Query(...),
    v2: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> ComparisonResponse:
    """Compare metrics between two model versions."""
    async def _latest_metrics(version: str) -> dict[str, float]:
        result = await db.execute(
            select(EvalRun)
            .where(
                EvalRun.model_name == model_name,
                EvalRun.model_version == version,
                EvalRun.status == "completed",
            )
            .order_by(EvalRun.started_at.desc())
            .limit(1)
        )
        run = result.scalar_one_or_none()
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No completed eval run for {model_name}:{version}",
            )
        await db.refresh(run, ["metrics"])
        return {m.metric_name: m.value for m in run.metrics}

    m1 = await _latest_metrics(v1)
    m2 = await _latest_metrics(v2)

    shared_keys = sorted(set(m1.keys()) & set(m2.keys()))
    comparisons: list[ComparisonItem] = []
    for key in shared_keys:
        val1, val2 = m1[key], m2[key]
        delta = val2 - val1
        pct = (delta / val1 * 100) if val1 != 0 else 0.0
        comparisons.append(
            ComparisonItem(
                metric=key,
                v1=round(val1, 4),
                v2=round(val2, 4),
                delta=round(delta, 4),
                pct_change=round(pct, 2),
            )
        )

    return ComparisonResponse(
        model_name=model_name,
        v1=v1,
        v2=v2,
        comparison=comparisons,
    )
