"""LLM recommendations: one generation per opportunity, scored candidates, expert feedback, implementations.

Revision ID: 0007_recommendations
Revises: 0006_classification_output
Create Date: 2026-11-20

recommendation_generations
    One row per opportunity and generator_version (re-running the same version is a no-op).
    evidence_pack is exactly what the LLM was given, so every cited id can be resolved later.
    status: succeeded | insufficient_evidence (no LLM call) | failed (cost kept, no candidates).
    output is the applied output, or the last rejected one on failure.

recommendation_candidates (existing)
    LLM rows are is_synthetic = false, linked to their generation and position in the LLM's list.
    evidence holds only the cited evidence ids; the facts live in the generation's evidence_pack.
    The synthetic-seed columns (topic, query_family_id, evidence_pattern, evidence_required) become
    optional, but stay required for synthetic rows.

recommendation_scores
    One immutable row per candidate and scoring_version (p1 = formula, later m1 = learned model).
    features are the scoring inputs, kept as the ML features.

recommendation_feedback
    Expert ratings, the ML labels. Immutable: a changed opinion is a new row, and
    latest_recommendation_feedback shows each reviewer's latest rating per candidate.
    shown_scoring_version / shown_rank record what the reviewer saw, to correct for position bias.

recommendation_implementations
    When a recommendation was actually carried out. Visibility before and after implemented_on is
    computed from observations later, not stored here.
"""

from __future__ import annotations

from alembic import op

revision = "0007_recommendations"
down_revision = "0006_classification_output"
branch_labels = None
depends_on = None

REASONS = ("not_actionable", "wrong_evidence", "wrong_category", "low_impact", "too_much_effort", "already_done",
           "other")
NEW_TABLES = ["recommendation_generations", "recommendation_scores", "recommendation_feedback",
              "recommendation_implementations"]
SEED_COLUMNS = ["topic", "query_family_id", "evidence_pattern", "evidence_required"]

UPGRADE = [
    r"""
    CREATE TABLE recommendation_generations (
        generation_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        opportunity_id       text NOT NULL CHECK (btrim(opportunity_id) <> ''),
        generator_version    text NOT NULL CHECK (generator_version ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'),
        target_brand         text NOT NULL REFERENCES brands (brand_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        status               text NOT NULL CHECK (status IN ('succeeded', 'insufficient_evidence', 'failed')),
        evidence_pack        jsonb NOT NULL CHECK (jsonb_typeof(evidence_pack) = 'object'),
        llm_provider         text NOT NULL,
        model_requested      text NOT NULL,
        model_id             text,
        prompt_version       text NOT NULL,
        schema_version       text NOT NULL,
        output               jsonb,
        insufficient_reason  text,
        error_type           text,
        error_message        text,
        validation_errors    jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(validation_errors) = 'array'),
        attempts             integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        input_tokens         integer CHECK (input_tokens >= 0),
        cached_input_tokens  integer CHECK (cached_input_tokens >= 0),
        output_tokens        integer CHECK (output_tokens >= 0),
        reasoning_tokens     integer CHECK (reasoning_tokens >= 0),
        latency_ms           integer CHECK (latency_ms >= 0),
        estimated_cost_usd   numeric(12, 8) CHECK (estimated_cost_usd >= 0),
        created_at           timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT recommendation_generations_one_per_version UNIQUE (opportunity_id, generator_version),
        CONSTRAINT recommendation_generations_succeeded_has_output CHECK (status <> 'succeeded' OR output IS NOT NULL),
        CONSTRAINT recommendation_generations_insufficient_has_reason
            CHECK (status <> 'insufficient_evidence' OR (insufficient_reason IS NOT NULL AND attempts = 0)),
        CONSTRAINT recommendation_generations_failed_has_error
            CHECK (status <> 'failed' OR error_type IS NOT NULL OR validation_errors <> '[]')
    )
    """,
    "CREATE INDEX recommendation_generations_target_brand_idx ON recommendation_generations (target_brand)",
    # --- LLM rows in recommendation_candidates -------------------------------
    "ALTER TABLE recommendation_candidates "
    + ", ".join(f"ALTER COLUMN {c} DROP NOT NULL" for c in SEED_COLUMNS),
    """
    ALTER TABLE recommendation_candidates
        ADD COLUMN generation_id    bigint REFERENCES recommendation_generations ON DELETE RESTRICT,
        ADD COLUMN position         smallint CHECK (position >= 0),
        ADD COLUMN title            text,
        ADD COLUMN expected_impact  text CHECK (expected_impact IN ('low', 'medium', 'high')),
        ADD COLUMN confidence       text CHECK (confidence IN ('low', 'medium', 'high')),
        ADD CONSTRAINT recommendation_candidates_one_per_position UNIQUE (generation_id, position),
        ADD CONSTRAINT recommendation_candidates_seed_columns
            CHECK (NOT is_synthetic OR (topic IS NOT NULL AND query_family_id IS NOT NULL
                                        AND evidence_pattern IS NOT NULL AND evidence_required IS NOT NULL)),
        ADD CONSTRAINT recommendation_candidates_generated_columns
            CHECK (is_synthetic OR (generation_id IS NOT NULL AND position IS NOT NULL AND title IS NOT NULL
                                    AND expected_impact IS NOT NULL AND confidence IS NOT NULL
                                    AND jsonb_typeof(evidence) = 'array')),
        ADD CONSTRAINT recommendation_candidates_synthetic_not_generated
            CHECK (NOT is_synthetic OR generation_id IS NULL)
    """,
    # --- scores ----------------------------------------------------------------
    r"""
    CREATE TABLE recommendation_scores (
        candidate_id     text NOT NULL REFERENCES recommendation_candidates ON UPDATE CASCADE ON DELETE CASCADE,
        scoring_version  text NOT NULL CHECK (scoring_version ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'),
        score            numeric(5, 2) NOT NULL CHECK (score BETWEEN 0 AND 100),
        band             text NOT NULL CHECK (band IN ('high', 'medium', 'low')),
        gap              numeric(6, 4) NOT NULL CHECK (gap >= 0),
        parts            jsonb NOT NULL CHECK (jsonb_typeof(parts) = 'object'),
        features         jsonb NOT NULL CHECK (jsonb_typeof(features) = 'object'),
        created_at       timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (candidate_id, scoring_version)
    )
    """,
    # --- expert feedback ---------------------------------------------------------
    rf"""
    CREATE TABLE recommendation_feedback (
        feedback_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        candidate_id           text NOT NULL REFERENCES recommendation_candidates ON UPDATE CASCADE ON DELETE CASCADE,
        reviewer               text NOT NULL DEFAULT 'owner' CHECK (reviewer ~ '^[a-z][a-z0-9_]{{0,31}}$'),
        useful                 boolean NOT NULL,
        rating                 smallint CHECK (rating BETWEEN 1 AND 5),
        reasons                jsonb NOT NULL DEFAULT '[]'
                               CHECK (jsonb_typeof(reasons) = 'array'
                                      AND reasons <@ '[{", ".join(f'"{r}"' for r in REASONS)}]'::jsonb),
        comment                text CHECK (comment IS NULL OR btrim(comment) <> ''),
        shown_scoring_version  text,
        shown_rank             integer CHECK (shown_rank >= 1),
        created_at             timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT recommendation_feedback_shown_score
            FOREIGN KEY (candidate_id, shown_scoring_version) REFERENCES recommendation_scores ON DELETE CASCADE,
        CONSTRAINT recommendation_feedback_rank_needs_version
            CHECK (shown_rank IS NULL OR shown_scoring_version IS NOT NULL)
    )
    """,
    "CREATE INDEX recommendation_feedback_candidate_idx ON recommendation_feedback (candidate_id)",
    """
    CREATE VIEW latest_recommendation_feedback WITH (security_invoker = true) AS
    SELECT DISTINCT ON (candidate_id, reviewer) *
    FROM recommendation_feedback
    ORDER BY candidate_id, reviewer, created_at DESC, feedback_id DESC
    """,
    # --- implementations -------------------------------------------------------
    r"""
    CREATE TABLE recommendation_implementations (
        implementation_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        candidate_id       text NOT NULL UNIQUE
                           REFERENCES recommendation_candidates ON UPDATE CASCADE ON DELETE CASCADE,
        implemented_on     date NOT NULL,
        recorded_by        text NOT NULL DEFAULT 'owner' CHECK (recorded_by ~ '^[a-z][a-z0-9_]{0,31}$'),
        notes              text CHECK (notes IS NULL OR btrim(notes) <> ''),
        created_at         timestamptz NOT NULL DEFAULT now(),
        updated_at         timestamptz NOT NULL DEFAULT now()
    )
    """,
    # --- triggers and Supabase API exposure ---------------------------------------
    *[f"CREATE TRIGGER {t}_reject_update BEFORE UPDATE ON {t} FOR EACH ROW EXECUTE FUNCTION reject_update()"
      for t in ("recommendation_generations", "recommendation_scores", "recommendation_feedback")],
    "CREATE TRIGGER recommendation_implementations_set_updated_at BEFORE UPDATE ON recommendation_implementations "
    "FOR EACH ROW EXECUTE FUNCTION set_updated_at()",
    *[f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY" for t in NEW_TABLES],
]

DOWNGRADE = [
    "DROP VIEW IF EXISTS latest_recommendation_feedback",
    "DROP TABLE IF EXISTS recommendation_implementations",
    "DROP TABLE IF EXISTS recommendation_feedback",
    "DROP TABLE IF EXISTS recommendation_scores",
    # Generated rows cannot satisfy the restored NOT NULL seed columns.
    "DELETE FROM recommendation_candidates WHERE NOT is_synthetic AND generation_id IS NOT NULL",
    """
    ALTER TABLE recommendation_candidates
        DROP CONSTRAINT IF EXISTS recommendation_candidates_synthetic_not_generated,
        DROP CONSTRAINT IF EXISTS recommendation_candidates_generated_columns,
        DROP CONSTRAINT IF EXISTS recommendation_candidates_seed_columns,
        DROP CONSTRAINT IF EXISTS recommendation_candidates_one_per_position,
        DROP COLUMN IF EXISTS confidence,
        DROP COLUMN IF EXISTS expected_impact,
        DROP COLUMN IF EXISTS title,
        DROP COLUMN IF EXISTS position,
        DROP COLUMN IF EXISTS generation_id
    """,
    "ALTER TABLE recommendation_candidates " + ", ".join(f"ALTER COLUMN {c} SET NOT NULL" for c in SEED_COLUMNS),
    "DROP TABLE IF EXISTS recommendation_generations",
]


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
