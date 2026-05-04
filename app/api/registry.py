"""Model registry API routes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Counter, Literal
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_api_key
from app.db.models import ModelVersion, DeploymentEvent, InferenceLog

router = APIRouter(prefix="/models")

# --- Helper functions ---
def validate_local_artifact_exists(artifact_uri: str) -> Path:
    """Validate that a local model artifact exists and is a file."""
    path = Path(artifact_uri)

    if not path.exists():
        raise ValueError(f"Artifact not found: {artifact_uri}")

    if not path.is_file():
        raise ValueError(f"Artifact is not a file: {artifact_uri}")

    return path


def calculate_sha256(path: Path) -> str:
    """Calculate SHA256 hash for a local model artifact."""
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * percentile)
    return sorted_values[index]
# --- Pydantic schemas ---


class RegisterModelRequest(BaseModel):
    """Request body for registering a new model version."""

    name: str
    version: str
    artifact_uri: str
    runtime_config: dict | None = None
    metadata_: dict | None = None
    # Optional fields for better model tracking and management
    framework: str | None = Field(default=None, examples=["sklearn"])
    task_type: str | None = Field(default=None, examples=["classification"])
    tags: list[str] | None = None


class ActivateModelRequest(BaseModel):
    """Request body for activating a model version."""

    version: str
    action: Literal["activate", "rollback"] = "activate"
    reason: str | None = Field(default=None, max_length=500)

class ModelVersionResponse(BaseModel):
    """Response schema for a model version record."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    version: str
    artifact_uri: str
    runtime_config: dict | None = None
    metadata_: dict | None = None
    is_active: bool
    created_at: datetime
    # Optional fields for better model tracking and management
    framework: str | None = None
    task_type: str | None = None
    status: str
    tags: list[str] | None = None
    artifact_hash: str | None = None

class DeploymentEventResponse(BaseModel):
    """Response schema for deployment event history."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    model_name: str
    from_version: str | None
    to_version: str
    action: str
    reason: str | None
    created_at: datetime

class ModelStatsResponse(BaseModel):
    """Response schema for model performance statistics."""
    model_name: str
    active_version: str | None = None

    total_predictions: int
    success_count: int
    error_count: int
    error_rate: float

    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_total_tokens: float | None = None

    top_error_types: dict[str, int]

# --- Routes ---
@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=ModelVersionResponse, dependencies=[Depends(require_api_key)],)
async def register_model(
    body: RegisterModelRequest, db: AsyncSession = Depends(get_db)
) -> ModelVersion:
    """Register a new model version."""
    # Validate artifact exists and calculate hash before creating DB record
    adapter_type = (body.runtime_config or {}).get("adapter_type", "sklearn")

    artifact_hash: str | None = None

    if adapter_type != "llm":
        try:
            artifact_path = validate_local_artifact_exists(body.artifact_uri)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        artifact_hash = calculate_sha256(artifact_path)

    mv = ModelVersion(
        name=body.name,
        version=body.version,
        artifact_uri=body.artifact_uri,
        runtime_config=body.runtime_config,
        metadata_=body.metadata_,
        framework=body.framework,
        task_type=body.task_type,
        tags=body.tags,
        status="registered",
        artifact_hash=artifact_hash,
    )
    db.add(mv)
    
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, 
            detail=f"Model {body.name}:{body.version} already exists",
            ) from exc
    await db.refresh(mv)
    return mv


@router.post("/{name}/activate", response_model=ModelVersionResponse, dependencies=[Depends(require_api_key)])
async def activate_model(
    name: str, body: ActivateModelRequest, db: AsyncSession = Depends(get_db)
) -> ModelVersion:
    """Activate a specific version, deactivating all others."""
    # Find target version to activate first
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == body.version)
    )
    mv = result.scalar_one_or_none()
    if mv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version {body.version} not found for model {name}",
        )
    # Find current active version (if any) to deactivate
    current_result = await db.execute(
        select(ModelVersion).where(ModelVersion.name == name, ModelVersion.is_active)
    )
    current_active = current_result.scalar_one_or_none()

    # Idempotent check: if the target version is already active, just return it without making changes
    if current_active is not None and current_active.version == mv.version:
        return mv
    
    # Deactivate current active version (if exists)
    await db.execute(
        update(ModelVersion)
        .where(ModelVersion.name == name)
        .values(is_active=False)
    )

    # Activate target version
    mv.is_active = True

    # Record deployment event
    event = DeploymentEvent(
        model_name=name,
        from_version=current_active.version if current_active else None,
        to_version=mv.version,
        action=body.action,
        reason=body.reason,
    )
    
    db.add(event)
    await db.commit()
    await db.refresh(mv)
    return mv

@router.get("/{name}/deployments", response_model=list[DeploymentEventResponse])
async def list_deployment_events(
    name: str, db: AsyncSession = Depends(get_db)
) -> list[DeploymentEvent]:
    """Return all deployment events for a model."""
    result = await db.execute(
        select(DeploymentEvent).where(DeploymentEvent.model_name == name).order_by(DeploymentEvent.created_at.desc())
    )
    rows = list(result.scalars().all())
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No deployment events found for model {name}",
        )
    return rows

@router.get("/{name}/stats", response_model=ModelStatsResponse)
async def get_model_stats(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> ModelStatsResponse:
    active_result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == name,
            ModelVersion.is_active == True,  # noqa: E712
        )
    )
    active = active_result.scalar_one_or_none()

    logs_result = await db.execute(
        select(InferenceLog).where(InferenceLog.model_name == name)
    )
    logs = list(logs_result.scalars().all())

    total = len(logs)
    success_count = sum(1 for log in logs if 200 <= log.status_code < 400)
    error_count = total - success_count
    error_rate = error_count / total if total else 0.0

    latencies = [log.latency_ms for log in logs]
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    p95_latency = _percentile(latencies, 0.95)

    error_counter = Counter(
        log.error_type for log in logs if log.error_type is not None
    )

    input_token_values = [
        log.input_tokens for log in logs if log.input_tokens is not None
    ]
    output_token_values = [
        log.output_tokens for log in logs if log.output_tokens is not None
    ]

    total_input_tokens = sum(input_token_values)
    total_output_tokens = sum(output_token_values)
    total_tokens = total_input_tokens + total_output_tokens

    token_log_count = len(
        [
            log
            for log in logs
            if log.input_tokens is not None or log.output_tokens is not None
        ]
    )

    avg_input_tokens = (
        total_input_tokens / token_log_count if token_log_count else None
    )
    avg_output_tokens = (
        total_output_tokens / token_log_count if token_log_count else None
    )
    avg_total_tokens = (
        total_tokens / token_log_count if token_log_count else None
    )

    return ModelStatsResponse(
        model_name=name,
        active_version=active.version if active else None,
        total_predictions=total,
        success_count=success_count,
        error_count=error_count,
        error_rate=round(error_rate, 4),
        avg_latency_ms=round(avg_latency, 2) if avg_latency is not None else None,
        p95_latency_ms=round(p95_latency, 2) if p95_latency is not None else None,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        avg_input_tokens=round(avg_input_tokens, 2)
        if avg_input_tokens is not None
        else None,
        avg_output_tokens=round(avg_output_tokens, 2)
        if avg_output_tokens is not None
        else None,
        avg_total_tokens=round(avg_total_tokens, 2)
        if avg_total_tokens is not None
        else None,
        top_error_types=dict(error_counter),
    )


@router.get("/{name}", response_model=list[ModelVersionResponse])
async def list_model_versions(
    name: str, db: AsyncSession = Depends(get_db)
) -> list[ModelVersion]:
    """Return all versions for a model, active first."""
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.name == name)
        .order_by(ModelVersion.is_active.desc(), ModelVersion.created_at.desc())
    )
    rows = list(result.scalars().all())
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No versions found for model {name}",
        )
    return rows


@router.get("/{name}/active", response_model=ModelVersionResponse)
async def get_active_version(
    name: str, db: AsyncSession = Depends(get_db)
) -> ModelVersion:
    """Return only the currently active version."""
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == name, ModelVersion.is_active == True  # noqa: E712
        )
    )
    mv = result.scalar_one_or_none()
    if mv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active version found for model {name}",
        )
    return mv
