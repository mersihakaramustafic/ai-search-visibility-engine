"""Prefix the usage columns of observations with collection_.

Revision ID: 0005_collection_column_prefix
Revises: 0004_llm_classification
Create Date: 2026-10-22

observations and observation_extractions both record tokens and latency, but
for different API calls: collection_* is the provider answering the query,
classification_* is the extraction model reading that answer.
"""

from __future__ import annotations

from alembic import op

revision = "0005_collection_column_prefix"
down_revision = "0004_llm_classification"
branch_labels = None
depends_on = None

COLUMNS = ["input_tokens", "output_tokens", "latency_ms"]


def upgrade() -> None:
    for column in COLUMNS:
        op.execute(f"ALTER TABLE observations RENAME COLUMN {column} TO collection_{column}")


def downgrade() -> None:
    for column in COLUMNS:
        op.execute(f"ALTER TABLE observations RENAME COLUMN collection_{column} TO {column}")
