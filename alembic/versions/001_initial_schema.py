"""Initial schema.

Revision ID: 001
Revises:
Create Date: 2026-03-29

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("artifact_uri", sa.String(1024), nullable=False),
        sa.Column("runtime_config", JSONB, nullable=True),
        sa.Column("metadata_", JSONB, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="uq_model_name_version"),
    )

    op.create_table(
        "inference_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False, index=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("status_code", sa.Integer, nullable=False),
        sa.Column("error_type", sa.String(255), nullable=True),
        sa.Column("input_tokens", sa.Integer, nullable=True),
        sa.Column("output_tokens", sa.Integer, nullable=True),
        sa.Column("schema_valid", sa.Boolean, nullable=True),
        sa.Column("tool_success", sa.Boolean, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "eval_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False, index=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("dataset_id", sa.String(255), nullable=False),
        sa.Column("dataset_hash", sa.String(64), nullable=False),
        sa.Column("git_commit", sa.String(64), nullable=True),
        sa.Column("config_snapshot", JSONB, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
    )

    op.create_table(
        "eval_metrics",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("eval_runs.id"), nullable=False, index=True),
        sa.Column("metric_name", sa.String(128), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "job_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("job_type", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("error_msg", sa.Text, nullable=True),
        sa.Column("result", JSONB, nullable=True),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("job_records")
    op.drop_table("eval_metrics")
    op.drop_table("eval_runs")
    op.drop_table("inference_logs")
    op.drop_table("model_versions")
