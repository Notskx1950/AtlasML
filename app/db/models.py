"""SQLAlchemy ORM models for AtlasML."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid, func, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class ModelVersion(Base):
    """Registered model version with artifact location and config."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
    )
    
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    runtime_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata_", JSON, nullable=True) # avoid conflict with SQLAlchemy reserved name 'metadata', it is used to store arbitrary user-defined metadata about the model version
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # newly added fields for tracking model usage and performance
    framework: Mapped[str | None] = mapped_column(String(64), nullable=True)
    task_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="registered")
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    artifact_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class InferenceLog(Base):
    """Log entry for a single inference request."""

    __tablename__ = "inference_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    schema_valid: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    tool_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvalRun(Base):
    """Record of an evaluation run."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="running"
    )

    metrics: Mapped[list[EvalMetric]] = relationship(
        "EvalMetric", back_populates="run", cascade="all, delete-orphan"
    )


class EvalMetric(Base):
    """Single metric value produced by an eval run."""

    __tablename__ = "eval_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("eval_runs.id"), nullable=False, index=True
    )
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped[EvalRun] = relationship("EvalRun", back_populates="metrics")


class JobRecord(Base):
    """Tracks the lifecycle of an async RQ job."""

    __tablename__ = "job_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="queued"
    )
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    enqueued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

class DeploymentEvent(Base):
    """Record of a model deployment event, such as activation or deactivation."""
    __tablename__ = "deployment_events"

    __table_args__ = (
        Index("ix_deployment_events_model_created_at", "model_name", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid,primary_key=True,default=uuid.uuid4,)
    model_name: Mapped[str] = mapped_column(String,nullable=False,index=True,)
    from_version: Mapped[str | None] = mapped_column(String,nullable=True,)
    to_version: Mapped[str] = mapped_column(String,nullable=False,)
    action: Mapped[str] = mapped_column(String,nullable=False,)
    reason: Mapped[str | None] = mapped_column(String,nullable=True,)
    created_at = mapped_column(DateTime(timezone=True),server_default=func.now(),nullable=False,)