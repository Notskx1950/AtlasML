"""Model registry API routes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.db.models import ModelVersion

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

# --- Routes ---


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=ModelVersionResponse)
async def register_model(
    body: RegisterModelRequest, db: AsyncSession = Depends(get_db)
) -> ModelVersion:
    """Register a new model version."""
    # Validate artifact exists and calculate hash before creating DB record
    try:
        artifact_path = validate_local_artifact_exists(body.artifact_uri)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
    await db.commit()
    await db.refresh(mv)
    return mv


@router.post("/{name}/activate", response_model=ModelVersionResponse)
async def activate_model(
    name: str, body: ActivateModelRequest, db: AsyncSession = Depends(get_db)
) -> ModelVersion:
    """Activate a specific version, deactivating all others."""
    # Deactivate all versions for this model
    await db.execute(
        update(ModelVersion)
        .where(ModelVersion.name == name)
        .values(is_active=False)
    )
    # Activate the requested version
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == name, ModelVersion.version == body.version
        )
    )
    mv = result.scalar_one_or_none()
    if mv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {body.version} not found for model {name}",
        )
    mv.is_active = True
    await db.commit()
    await db.refresh(mv)
    return mv


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
