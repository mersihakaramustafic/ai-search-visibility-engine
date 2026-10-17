"""Initial schema: reference data, collection plan, observations, recommendations.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-10-15

Layers:
    reference     brands, queries                       (authored, imported from CSV)
    plan          collection_requests                   (planned, imported from CSV)
    collected     collection_runs, observations         (written by future collectors)
    derived       brand_mentions, citations             (written by future extractors)
    recommend     recommendation_candidates             (synthetic seeds now, evidence-backed later)

Allowed values are CHECK constraints rather than PostgreSQL enum types, so a
later migration can widen them with a plain DROP/ADD CONSTRAINT.

Every table has row level security enabled with no policies. In Supabase that
blocks the public anon/authenticated API roles; the postgres role used for
migrations and imports owns the tables and is unaffected.
"""

from __future__ import annotations

from alembic import op

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None

TABLES = [
    "brands",
    "queries",
    "collection_runs",
    "collection_requests",
    "observations",
    "brand_mentions",
    "citations",
    "recommendation_candidates",
]
TABLES_WITH_UPDATED_AT = [
    "brands",
    "queries",
    "collection_runs",
    "collection_requests",
    "recommendation_candidates",
]

UPGRADE = [
    # --- helpers -------------------------------------------------------------
    """
    CREATE FUNCTION set_updated_at() RETURNS trigger
    LANGUAGE plpgsql SET search_path = '' AS $$
    BEGIN
        NEW.updated_at = now();
        RETURN NEW;
    END
    $$
    """,
    """
    CREATE FUNCTION reject_update() RETURNS trigger
    LANGUAGE plpgsql SET search_path = '' AS $$
    BEGIN
        -- No '%' format placeholder: it would need escaping through the driver.
        RAISE EXCEPTION USING MESSAGE = TG_TABLE_NAME || ' rows are immutable; insert a new row instead';
    END
    $$
    """,
    # --- reference data ------------------------------------------------------
    r"""
    CREATE TABLE brands (
        brand_id          text PRIMARY KEY CHECK (brand_id ~ '^[a-z][a-z0-9_]*$'),
        brand_name        text NOT NULL UNIQUE CHECK (btrim(brand_name) <> ''),
        canonical_domain  text NOT NULL UNIQUE
                          CHECK (canonical_domain ~ '^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$'),
        aliases           jsonb NOT NULL
                          CHECK (jsonb_typeof(aliases) = 'array' AND jsonb_array_length(aliases) > 0),
        description       text NOT NULL,
        product_category  text NOT NULL
                          CHECK (product_category IN ('crm', 'crm_work_management', 'customer_service_adjacent')),
        is_primary_brand  boolean NOT NULL DEFAULT false,
        created_at        timestamptz NOT NULL DEFAULT now(),
        updated_at        timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE queries (
        query_id             text PRIMARY KEY CHECK (query_id ~ '^Q[0-9]{3,}$'),
        query_text           text NOT NULL UNIQUE CHECK (btrim(query_text) <> ''),
        intent               text NOT NULL CHECK (intent IN (
                                 'category_discovery', 'use_case', 'brand_comparison',
                                 'features_integrations', 'pricing_purchase')),
        topic                text NOT NULL,
        target_audience      text NOT NULL,
        language             text NOT NULL,
        market               text NOT NULL,
        contains_brand_name  boolean NOT NULL,
        named_brands         jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(named_brands) = 'array'),
        query_family_id      text NOT NULL CHECK (query_family_id ~ '^fam_[a-z0-9_]+$'),
        created_at           timestamptz NOT NULL DEFAULT now(),
        updated_at           timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT queries_brand_flag_matches_named_brands
            CHECK (contains_brand_name = (jsonb_array_length(named_brands) > 0))
    )
    """,
    "CREATE INDEX queries_intent_idx ON queries (intent)",
    "CREATE INDEX queries_query_family_id_idx ON queries (query_family_id)",
    # --- collection plan and runs ------------------------------------------
    """
    CREATE TABLE collection_runs (
        collection_run_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        run_label          text NOT NULL UNIQUE,
        status             text NOT NULL DEFAULT 'pending'
                           CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
        collector_config   jsonb NOT NULL DEFAULT '{}'::jsonb,
        code_version       text,
        notes              text,
        started_at         timestamptz,
        finished_at        timestamptz,
        created_at         timestamptz NOT NULL DEFAULT now(),
        updated_at         timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT collection_runs_finished_after_started
            CHECK (finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at)
    )
    """,
    "CREATE INDEX collection_runs_status_idx ON collection_runs (status)",
    """
    CREATE TABLE collection_requests (
        request_id      text PRIMARY KEY,
        query_id        text NOT NULL REFERENCES queries (query_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        provider        text NOT NULL CHECK (provider ~ '^[a-z][a-z0-9_]*$'),
        run_number      integer NOT NULL CHECK (run_number >= 1),
        market          text NOT NULL,
        language        text NOT NULL,
        search_enabled  boolean NOT NULL,
        status          text NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'skipped')),
        created_at      timestamptz NOT NULL DEFAULT now(),
        updated_at      timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT collection_requests_query_provider_run_key UNIQUE (query_id, provider, run_number),
        CONSTRAINT collection_requests_request_id_format
            CHECK (request_id = 'REQ-' || query_id || '-' || provider || '-R' || run_number::text)
    )
    """,
    "CREATE INDEX collection_requests_provider_status_idx ON collection_requests (provider, status)",
    "CREATE INDEX collection_requests_status_idx ON collection_requests (status)",
    # --- collected observations (one immutable row per API attempt) ---------
    """
    CREATE TABLE observations (
        observation_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        request_id          text NOT NULL REFERENCES collection_requests (request_id)
                            ON UPDATE CASCADE ON DELETE RESTRICT,
        collection_run_id   bigint REFERENCES collection_runs (collection_run_id) ON DELETE RESTRICT,
        attempt_number      integer NOT NULL DEFAULT 1 CHECK (attempt_number >= 1),
        provider            text NOT NULL,
        model_id            text,
        request_params      jsonb NOT NULL DEFAULT '{}'::jsonb,
        search_config       jsonb NOT NULL DEFAULT '{}'::jsonb,
        status              text NOT NULL CHECK (status IN ('succeeded', 'failed')),
        response_text       text,
        raw_response        jsonb,
        grounding_metadata  jsonb,
        error_type          text,
        error_message       text,
        latency_ms          integer CHECK (latency_ms >= 0),
        input_tokens        integer CHECK (input_tokens >= 0),
        output_tokens       integer CHECK (output_tokens >= 0),
        requested_at        timestamptz NOT NULL,
        responded_at        timestamptz,
        created_at          timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT observations_request_attempt_key UNIQUE (request_id, attempt_number),
        CONSTRAINT observations_success_has_response CHECK (status <> 'succeeded' OR response_text IS NOT NULL),
        CONSTRAINT observations_failure_has_error CHECK (status <> 'failed' OR error_message IS NOT NULL),
        CONSTRAINT observations_responded_after_requested
            CHECK (responded_at IS NULL OR responded_at >= requested_at)
    )
    """,
    "CREATE INDEX observations_collection_run_id_idx ON observations (collection_run_id)",
    "CREATE INDEX observations_provider_model_idx ON observations (provider, model_id)",
    "CREATE INDEX observations_status_idx ON observations (status)",
    # --- derived from observations ------------------------------------------
    """
    CREATE TABLE brand_mentions (
        brand_mention_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        observation_id      bigint NOT NULL REFERENCES observations (observation_id) ON DELETE CASCADE,
        brand_id            text NOT NULL REFERENCES brands (brand_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        extractor_version   text NOT NULL,
        mention_count       integer NOT NULL CHECK (mention_count >= 1),
        first_mention_rank  integer CHECK (first_mention_rank >= 1),
        first_char_offset   integer CHECK (first_char_offset >= 0),
        matched_alias       text,
        context_snippet     text,
        created_at          timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT brand_mentions_observation_brand_extractor_key
            UNIQUE (observation_id, brand_id, extractor_version)
    )
    """,
    "CREATE INDEX brand_mentions_brand_id_idx ON brand_mentions (brand_id)",
    """
    CREATE TABLE citations (
        citation_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        observation_id     bigint NOT NULL REFERENCES observations (observation_id) ON DELETE CASCADE,
        extractor_version  text NOT NULL,
        citation_position  integer NOT NULL CHECK (citation_position >= 1),
        url                text NOT NULL,
        domain             text NOT NULL,
        title              text,
        brand_id           text REFERENCES brands (brand_id) ON UPDATE CASCADE ON DELETE SET NULL,
        created_at         timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT citations_observation_extractor_position_key
            UNIQUE (observation_id, extractor_version, citation_position)
    )
    """,
    "CREATE INDEX citations_domain_idx ON citations (domain)",
    "CREATE INDEX citations_brand_id_idx ON citations (brand_id)",
    # --- recommendations ------------------------------------------------------
    # Synthetic seed rows (is_synthetic = true) must have no evidence and an
    # "Untested hypothesis:" prefix; evidence-backed rows must carry evidence.
    """
    CREATE TABLE recommendation_candidates (
        candidate_id       text PRIMARY KEY CHECK (btrim(candidate_id) <> ''),
        target_brand       text NOT NULL REFERENCES brands (brand_id) ON UPDATE CASCADE ON DELETE RESTRICT,
        action_type        text NOT NULL CHECK (action_type IN (
                               'owned_content_optimization', 'editorial_outreach',
                               'community_visibility', 'directory_listing')),
        topic              text NOT NULL,
        query_family_id    text NOT NULL CHECK (query_family_id ~ '^fam_[a-z0-9_]+$'),
        evidence_pattern   text NOT NULL CHECK (evidence_pattern IN (
                               'competitor_coverage_gap', 'cited_comparison_page',
                               'missing_directory_listing', 'recurring_user_question')),
        suggested_action   text NOT NULL,
        evidence_required  text NOT NULL,
        estimated_effort   text NOT NULL CHECK (estimated_effort IN ('low', 'medium', 'high')),
        hypothesis         text NOT NULL,
        is_synthetic       boolean NOT NULL,
        evidence           jsonb,
        status             text NOT NULL DEFAULT 'proposed'
                           CHECK (status IN ('proposed', 'under_review', 'accepted', 'rejected', 'implemented')),
        created_at         timestamptz NOT NULL DEFAULT now(),
        updated_at         timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT recommendation_candidates_evidence_matches_provenance
            CHECK (is_synthetic = (evidence IS NULL)),
        CONSTRAINT recommendation_candidates_synthetic_shape CHECK (
            NOT is_synthetic
            OR (candidate_id ~ '^REC[0-9]{3,}$' AND starts_with(hypothesis, 'Untested hypothesis:'))
        )
    )
    """,
    "CREATE INDEX recommendation_candidates_target_brand_idx ON recommendation_candidates (target_brand)",
    "CREATE INDEX recommendation_candidates_is_synthetic_idx ON recommendation_candidates (is_synthetic)",
    """
    CREATE VIEW synthetic_recommendation_candidates WITH (security_invoker = true) AS
    SELECT * FROM recommendation_candidates WHERE is_synthetic
    """,
    """
    CREATE VIEW evidence_backed_recommendation_candidates WITH (security_invoker = true) AS
    SELECT * FROM recommendation_candidates WHERE NOT is_synthetic
    """,
    # --- triggers -------------------------------------------------------------
    *[
        f"CREATE TRIGGER {table}_set_updated_at BEFORE UPDATE ON {table} "
        f"FOR EACH ROW EXECUTE FUNCTION set_updated_at()"
        for table in TABLES_WITH_UPDATED_AT
    ],
    "CREATE TRIGGER observations_reject_update BEFORE UPDATE ON observations "
    "FOR EACH ROW EXECUTE FUNCTION reject_update()",
    # --- Supabase API exposure -----------------------------------------------
    *[f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" for table in TABLES],
    "ALTER TABLE alembic_version ENABLE ROW LEVEL SECURITY",
]

DOWNGRADE = [
    "DROP VIEW IF EXISTS evidence_backed_recommendation_candidates",
    "DROP VIEW IF EXISTS synthetic_recommendation_candidates",
    *[f"DROP TABLE IF EXISTS {table}" for table in reversed(TABLES)],
    "DROP FUNCTION IF EXISTS reject_update()",
    "DROP FUNCTION IF EXISTS set_updated_at()",
]


def upgrade() -> None:
    for statement in UPGRADE:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE:
        op.execute(statement)
