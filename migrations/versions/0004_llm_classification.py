"""LLM classification: provenance and cost per extraction, labels per brand mention.

Revision ID: 0004_llm_classification
Revises: 0003_rules_extraction
Create Date: 2026-10-22

strategy says whether an extraction used only rules or rules plus the LLM.
classification_status is always NULL for rules_only. For rules_and_llm it is:
    not_needed  nothing to classify (no brands, no candidates), so no call was made
    succeeded   the output passed validation and was applied
    failed      API errors or invalid output on every attempt; nothing was stored but the cost
    NULL        the rules step already failed, so the LLM was never reached

rejected_candidates keeps the ambiguous words the LLM decided were not the brand,
with its quote. Accepted ones become brand_mentions rows; rejected ones would
otherwise leave no trace. NULL when no LLM output was applied.

Existing rows were all produced by the rules-only extractor, so the default fits them.
"""

from __future__ import annotations

from alembic import op

revision = "0004_llm_classification"
down_revision = "0003_rules_extraction"
branch_labels = None
depends_on = None

UPGRADE = [
    """
    ALTER TABLE observation_extractions
        ADD COLUMN strategy text NOT NULL DEFAULT 'rules_only' CHECK (strategy IN ('rules_only', 'rules_and_llm')),
        ADD COLUMN classification_status text CHECK (classification_status IN ('not_needed', 'succeeded', 'failed')),
        ADD COLUMN classification_model_requested text,
        ADD COLUMN classification_model_id text,
        ADD COLUMN prompt_version text,
        ADD COLUMN schema_version text,
        ADD COLUMN rejected_candidates jsonb CHECK (jsonb_typeof(rejected_candidates) = 'array'),
        ADD COLUMN classification_attempts integer NOT NULL DEFAULT 0 CHECK (classification_attempts >= 0),
        ADD COLUMN classification_input_tokens integer CHECK (classification_input_tokens >= 0),
        ADD COLUMN classification_cached_input_tokens integer CHECK (classification_cached_input_tokens >= 0),
        ADD COLUMN classification_output_tokens integer CHECK (classification_output_tokens >= 0),
        ADD COLUMN classification_reasoning_tokens integer CHECK (classification_reasoning_tokens >= 0),
        ADD COLUMN classification_estimated_cost_usd numeric(12, 8) CHECK (classification_estimated_cost_usd >= 0),
        ADD COLUMN classification_latency_ms integer CHECK (classification_latency_ms >= 0),
        ADD COLUMN target_brand_recommended boolean,
        ADD CONSTRAINT observation_extractions_classification_status_matches_strategy
            CHECK (strategy = 'rules_and_llm' OR classification_status IS NULL),
        ADD CONSTRAINT observation_extractions_classification_call_has_model
            CHECK (classification_attempts = 0 OR classification_model_requested IS NOT NULL)
    """,
    """
    ALTER TABLE brand_mentions
        ADD COLUMN sentiment text CHECK (sentiment IN ('positive', 'neutral', 'negative')),
        ADD COLUMN recommendation text CHECK (recommendation IN ('primary', 'conditional', 'none')),
        ADD COLUMN classification_evidence text
    """,
]

DOWNGRADE = [
    """
    ALTER TABLE brand_mentions
        DROP COLUMN IF EXISTS sentiment,
        DROP COLUMN IF EXISTS recommendation,
        DROP COLUMN IF EXISTS classification_evidence
    """,
    """
    ALTER TABLE observation_extractions
        DROP CONSTRAINT IF EXISTS observation_extractions_classification_call_has_model,
        DROP CONSTRAINT IF EXISTS observation_extractions_classification_status_matches_strategy,
        DROP COLUMN IF EXISTS target_brand_recommended,
        DROP COLUMN IF EXISTS classification_latency_ms,
        DROP COLUMN IF EXISTS classification_estimated_cost_usd,
        DROP COLUMN IF EXISTS classification_reasoning_tokens,
        DROP COLUMN IF EXISTS classification_output_tokens,
        DROP COLUMN IF EXISTS classification_cached_input_tokens,
        DROP COLUMN IF EXISTS classification_input_tokens,
        DROP COLUMN IF EXISTS classification_attempts,
        DROP COLUMN IF EXISTS rejected_candidates,
        DROP COLUMN IF EXISTS schema_version,
        DROP COLUMN IF EXISTS prompt_version,
        DROP COLUMN IF EXISTS classification_model_id,
        DROP COLUMN IF EXISTS classification_model_requested,
        DROP COLUMN IF EXISTS classification_status,
        DROP COLUMN IF EXISTS strategy
    """,
]


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
