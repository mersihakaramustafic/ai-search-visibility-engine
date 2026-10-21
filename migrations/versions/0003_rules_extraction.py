"""Rules-only extraction: one row per processed observation, plus provenance on mentions and citations.

Revision ID: 0003_rules_extraction
Revises: 0002_one_success_per_request
Create Date: 2026-10-21

observation_extractions records that an observation was processed by an
extractor version, even when it has no mentions or citations, and whether the
result passed validation. brand_mentions and citations rows hang off it, so
deleting an extraction removes everything it produced, which is how a re-run
of the same version replaces its rows.

The LLM classification columns are added by a later migration.
"""

from __future__ import annotations

from alembic import op

revision = "0003_rules_extraction"
down_revision = "0002_one_success_per_request"
branch_labels = None
depends_on = None

UPGRADE = [
    """
    CREATE TABLE observation_extractions (
        extraction_id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        observation_id               bigint NOT NULL REFERENCES observations (observation_id) ON DELETE CASCADE,
        extractor_version            text NOT NULL CHECK (btrim(extractor_version) <> ''),
        status                       text NOT NULL CHECK (status IN ('succeeded', 'failed')),
        validation_status            text CHECK (validation_status IN ('valid', 'valid_with_warnings', 'invalid')),
        validation_issues            jsonb NOT NULL DEFAULT '[]'::jsonb
                                     CHECK (jsonb_typeof(validation_issues) = 'array'),
        target_brand_mentioned       boolean,
        competitor_brands_mentioned  integer CHECK (competitor_brands_mentioned >= 0),
        candidate_count              integer CHECK (candidate_count >= 0),
        error_type                   text,
        error_message                text,
        created_at                   timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT observation_extractions_observation_version_key UNIQUE (observation_id, extractor_version),
        CONSTRAINT observation_extractions_validated_when_succeeded
            CHECK ((status = 'succeeded') = (validation_status IS NOT NULL)),
        CONSTRAINT observation_extractions_failure_has_error CHECK (status <> 'failed' OR error_message IS NOT NULL)
    )
    """,
    "CREATE INDEX observation_extractions_status_idx ON observation_extractions (status, validation_status)",
    "ALTER TABLE observation_extractions ENABLE ROW LEVEL SECURITY",
    # --- brand_mentions ---------------------------------------------------------
    """
    ALTER TABLE brand_mentions
        ADD COLUMN match_method text NOT NULL CHECK (match_method IN ('alias', 'llm_disambiguated')),
        ADD CONSTRAINT brand_mentions_extraction_fkey FOREIGN KEY (observation_id, extractor_version)
            REFERENCES observation_extractions (observation_id, extractor_version) ON DELETE CASCADE
    """,
    # --- citations --------------------------------------------------------------
    """
    ALTER TABLE citations
        ALTER COLUMN domain DROP NOT NULL,
        ADD COLUMN citation_source text NOT NULL CHECK (citation_source IN ('provider_metadata', 'response_text')),
        ADD COLUMN provider_format text NOT NULL
            CHECK (provider_format IN ('openai_url_citation', 'gemini_grounding_chunk', 'text_url')),
        ADD COLUMN metadata jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
        ADD COLUMN is_target_brand_domain boolean,
        ADD CONSTRAINT citations_extraction_fkey FOREIGN KEY (observation_id, extractor_version)
            REFERENCES observation_extractions (observation_id, extractor_version) ON DELETE CASCADE
    """,
    "CREATE INDEX citations_citation_source_idx ON citations (citation_source)",
]

DOWNGRADE = [
    "DROP INDEX IF EXISTS citations_citation_source_idx",
    "ALTER TABLE citations DROP CONSTRAINT IF EXISTS citations_extraction_fkey",
    "ALTER TABLE brand_mentions DROP CONSTRAINT IF EXISTS brand_mentions_extraction_fkey",
    # Citations without a domain could not exist before this migration; they can be re-extracted.
    "DELETE FROM citations WHERE domain IS NULL",
    """
    ALTER TABLE citations
        DROP COLUMN IF EXISTS citation_source,
        DROP COLUMN IF EXISTS provider_format,
        DROP COLUMN IF EXISTS metadata,
        DROP COLUMN IF EXISTS is_target_brand_domain,
        ALTER COLUMN domain SET NOT NULL
    """,
    "ALTER TABLE brand_mentions DROP COLUMN IF EXISTS match_method",
    "DROP TABLE IF EXISTS observation_extractions",
]


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
