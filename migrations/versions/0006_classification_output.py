"""Keep the classification model's output on each extraction, replacing rejected_candidates.

Revision ID: 0006_classification_output
Revises: 0005_collection_column_prefix
Create Date: 2026-10-22

On success it is the validated output that was applied; on failure it is the
last rejected output (a JSON string if it was not valid JSON), so a failure can
be inspected without calling the model again. Rejected candidates are part of
it (is_brand false), so the separate rejected_candidates column is dropped.
"""

from __future__ import annotations

from alembic import op

revision = "0006_classification_output"
down_revision = "0005_collection_column_prefix"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE observation_extractions ADD COLUMN classification_output jsonb, "
               "DROP COLUMN rejected_candidates")


def downgrade() -> None:
    op.execute("ALTER TABLE observation_extractions DROP COLUMN IF EXISTS classification_output, "
               "ADD COLUMN rejected_candidates jsonb CHECK (jsonb_typeof(rejected_candidates) = 'array')")
