"""At most one succeeded observation per collection request.

Revision ID: 0002_one_success_per_request
Revises: 0001_initial_schema
Create Date: 2026-10-17

Failed attempts stay as separate rows (one per API call), but a request can
only ever be answered once. This makes a retried or re-run request unable to
create a duplicate successful observation, even if two collectors overlap.
"""

from __future__ import annotations

from alembic import op

revision = "0002_one_success_per_request"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE UNIQUE INDEX observations_one_success_per_request "
        "ON observations (request_id) WHERE status = 'succeeded'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS observations_one_success_per_request")
