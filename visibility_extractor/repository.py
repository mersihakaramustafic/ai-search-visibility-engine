"""Database access for the extractor: read brands and observations, store extractions."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine

from visibility_extractor.matching import Brand
from visibility_extractor.pipeline import Extraction, LLMRun, Observation

_INSERT_EXTRACTION = text("""
    INSERT INTO observation_extractions (
        observation_id, extractor_version, status, validation_status, validation_issues,
        target_brand_mentioned, competitor_brands_mentioned, candidate_count, error_type, error_message,
        strategy, target_brand_recommended, prompt_version, schema_version,
        classification_output, classification_status, classification_model_requested, classification_model_id, classification_attempts,
        classification_input_tokens, classification_cached_input_tokens, classification_output_tokens,
        classification_reasoning_tokens, classification_estimated_cost_usd, classification_latency_ms)
    VALUES (
        :observation_id, :extractor_version, :status, :validation_status, :validation_issues,
        :target_brand_mentioned, :competitor_brands_mentioned, :candidate_count, :error_type, :error_message,
        :strategy, :target_brand_recommended, :prompt_version, :schema_version,
        :classification_output, :classification_status, :classification_model_requested, :classification_model_id, :classification_attempts,
        :classification_input_tokens, :classification_cached_input_tokens, :classification_output_tokens,
        :classification_reasoning_tokens, :classification_estimated_cost_usd, :classification_latency_ms)
""").bindparams(*(bindparam(c, type_=JSONB(none_as_null=True))
                  for c in ("validation_issues", "classification_output")))

_INSERT_MENTION = text("""
    INSERT INTO brand_mentions (
        observation_id, brand_id, extractor_version, mention_count, first_mention_rank,
        first_char_offset, matched_alias, context_snippet, match_method, sentiment, recommendation,
        classification_evidence)
    VALUES (
        :observation_id, :brand_id, :extractor_version, :mention_count, :first_mention_rank,
        :first_char_offset, :matched_alias, :context_snippet, :match_method, :sentiment, :recommendation,
        :classification_evidence)
""")

_INSERT_CITATION = text("""
    INSERT INTO citations (
        observation_id, extractor_version, citation_position, url, domain, title, brand_id,
        citation_source, provider_format, metadata, is_target_brand_domain)
    VALUES (
        :observation_id, :extractor_version, :citation_position, :url, :domain, :title, :brand_id,
        :citation_source, :provider_format, :metadata, :is_target_brand_domain)
""").bindparams(bindparam("metadata", type_=JSONB))


def _classification_columns(run: LLMRun | None) -> dict[str, Any]:
    r = run or LLMRun(None, None, None, None)
    return {"prompt_version": r.prompt_version, "schema_version": r.schema_version,
            "classification_output": r.output,
            "classification_status": r.status, "classification_model_requested": r.model_requested,
            "classification_model_id": r.model_id, "classification_attempts": r.attempts,
            "classification_input_tokens": r.input_tokens,
            "classification_cached_input_tokens": r.cached_input_tokens,
            "classification_output_tokens": r.output_tokens,
            "classification_reasoning_tokens": r.reasoning_tokens,
            "classification_estimated_cost_usd": r.estimated_cost_usd,
            "classification_latency_ms": r.latency_ms}


class ExtractionRepository:
    def __init__(self, engine: Engine, begin: Callable[[], AbstractContextManager[Connection]] | None = None):
        self._engine = engine
        self._begin = begin or engine.begin

    def load_brands(self) -> list[Brand]:
        sql = "SELECT brand_id, brand_name, canonical_domain, aliases, is_primary_brand FROM brands ORDER BY brand_id"
        with self._begin() as conn:
            return [Brand(r.brand_id, r.brand_name, r.canonical_domain, tuple(r.aliases), r.is_primary_brand)
                    for r in conn.execute(text(sql))]

    def select_observations(self, extractor_version: str, providers: list[str] | None = None,
                            observation_ids: list[int] | None = None, limit: int | None = None,
                            reprocess: bool = False, retry_failed: bool = False) -> list[Observation]:
        """Succeeded observations, by default only those this version has not processed yet.

        retry_failed also selects those whose extraction by this version failed.
        """
        sql = """
            SELECT o.observation_id, o.provider, o.response_text, o.grounding_metadata, q.query_text
            FROM observations o
            JOIN collection_requests r ON r.request_id = o.request_id
            JOIN queries q ON q.query_id = r.query_id
            WHERE o.status = 'succeeded'
              AND (CAST(:providers AS text[]) IS NULL OR o.provider = ANY(CAST(:providers AS text[])))
              AND (CAST(:ids AS bigint[]) IS NULL OR o.observation_id = ANY(CAST(:ids AS bigint[])))
              AND (:reprocess OR NOT EXISTS (
                  SELECT 1 FROM observation_extractions e
                  WHERE e.observation_id = o.observation_id AND e.extractor_version = :version
                    AND NOT (:retry_failed AND e.status = 'failed')))
            ORDER BY o.observation_id
            LIMIT :limit
        """
        with self._begin() as conn:
            rows = conn.execute(text(sql), {"version": extractor_version, "providers": providers,
                                            "ids": observation_ids, "limit": limit, "reprocess": reprocess,
                                            "retry_failed": retry_failed})
            return [Observation(**r._mapping) for r in rows]

    def pilot_ids(self, providers: list[str], per_provider: int) -> list[int]:
        """A fixed, spread-out sample: the same observations every time, whatever has been processed."""
        sql = """
            SELECT observation_id FROM observations
            WHERE status = 'succeeded' AND provider = :provider
            ORDER BY md5(observation_id::text)
            LIMIT :n
        """
        with self._begin() as conn:
            return sorted(i for p in providers
                          for i in conn.execute(text(sql), {"provider": p, "n": per_provider}).scalars())

    def save(self, extraction: Extraction) -> None:
        e = extraction
        key = {"observation_id": e.observation_id, "extractor_version": e.extractor_version}
        with self._begin() as conn:
            conn.execute(text("DELETE FROM observation_extractions "
                              "WHERE observation_id = :observation_id AND extractor_version = :extractor_version"),
                         key)
            conn.execute(_INSERT_EXTRACTION, {
                **key, "status": e.status, "validation_status": e.validation_status,
                "validation_issues": e.validation_issues, "target_brand_mentioned": e.target_brand_mentioned,
                "competitor_brands_mentioned": e.competitor_brands_mentioned,
                "candidate_count": e.candidate_count, "error_type": e.error_type, "error_message": e.error_message,
                "strategy": e.strategy, "target_brand_recommended": e.target_brand_recommended,
                **_classification_columns(e.llm),
            })
            if e.mentions:
                conn.execute(_INSERT_MENTION, [{**key, **vars(m)} for m in e.mentions])
            if e.citations:
                conn.execute(_INSERT_CITATION, [{**key, **vars(c)} for c in e.citations])

    def extraction_summary(self, extractor_version: str) -> dict[str, Any]:
        sql = """
            SELECT
                (SELECT count(*) FROM observation_extractions WHERE extractor_version = :v) AS extractions,
                (SELECT count(*) FROM observation_extractions
                 WHERE extractor_version = :v AND status = 'failed') AS failed,
                (SELECT count(*) FROM observation_extractions
                 WHERE extractor_version = :v AND validation_status = 'invalid') AS invalid,
                (SELECT count(*) FROM brand_mentions WHERE extractor_version = :v) AS brand_mentions,
                (SELECT count(*) FROM citations WHERE extractor_version = :v) AS citations,
                (SELECT coalesce(sum(classification_estimated_cost_usd), 0) FROM observation_extractions
                 WHERE extractor_version = :v) AS classification_cost_usd
        """
        with self._begin() as conn:
            return dict(conn.execute(text(sql), {"v": extractor_version}).one()._mapping)
