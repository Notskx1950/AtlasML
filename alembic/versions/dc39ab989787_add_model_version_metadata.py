"""add model version metadata

Revision ID: dc39ab989787
Revises: 001
Create Date: 2026-04-28 14:47:26.700251

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'dc39ab989787'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column("framework", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("task_type", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="registered", # To keep existing rows consistent, we set a default value for the new status column
        ),
    )
    op.add_column(
        "model_versions",
        sa.Column("tags", sa.JSON(), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("artifact_hash", sa.String(length=128), nullable=True),
    )

    # Optional: remove server default after backfilling existing rows
    op.alter_column("model_versions", "status", server_default=None)


def downgrade() -> None:
    op.drop_column("model_versions", "artifact_hash")
    op.drop_column("model_versions", "tags")
    op.drop_column("model_versions", "status")
    op.drop_column("model_versions", "task_type")
    op.drop_column("model_versions", "framework")