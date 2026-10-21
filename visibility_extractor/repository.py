"""Database access for the extractor: read brands and observations, store extractions."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine

from visibility_extractor.matching import Brand
from visibility_extractor.pipeline import Extraction, Observation

_INSERT_EXTRACTION = text("""
    INSERT INTO observation_extractions (
        observation_id, extractor_version, status, validation_status, validation_issues,
        target_brand_mentioned, competitor_brands_mentioned, candidate_count, error_type, error_message)
    VALUES (
        :observation_id, :extractor_version, :status, :validation_status, :validation_issues,
        :target_brand_mentioned, :competitor_brands_mentioned, :candidate_count, :error_type, :error_message)
""").bindparams(bindparam("validation_issues", type_=JSONB))

_INSERT_MENTION = text("""
    INSERT INTO brand_mentions (
        observation_id, brand_id, extractor_version, mention_count, first_mention_rank,
        first_char_offset, matched_alias, context_snippet, match_method)
    VALUES (
        :observation_id, :brand_id, :extractor_version, :mention_count, :first_mention_rank,
        :first_char_offset, :matched_alias, :context_snippet, :match_method)
""")

_INSERT_CITATION = text("""
    INSERT INTO citations (
        observation_id, extractor_version, citation_position, url, domain, title, brand_id,
        citation_source, provider_format, metadata, is_target_brand_domain)
    VALUES (
        :observation_id, :extractor_version, :citation_position, :url, :domain, :title, :brand_id,
        :citation_source, :provider_format, :metadata, :is_target_brand_domain)
""").bindparams(bindparam("metadata", type_=JSONB))


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
                            reprocess: bool = False) -> list[Observation]:
        """Succeeded observations, by default only those this version has not processed yet."""
        sql = """
            SELECT o.observation_id, o.provider, o.response_text, o.grounding_metadata
            FROM observations o
            WHERE o.status = 'succeeded'
              AND (CAST(:providers AS text[]) IS NULL OR o.provider = ANY(CAST(:providers AS text[])))
              AND (CAST(:ids AS bigint[]) IS NULL OR o.observation_id = ANY(CAST(:ids AS bigint[])))
              AND (:reprocess OR NOT EXISTS (
                  SELECT 1 FROM observation_extractions e
                  WHERE e.observation_id = o.observation_id AND e.extractor_version = :version))
            ORDER BY o.observation_id
            LIMIT :limit
        """
        with self._begin() as conn:
            rows = conn.execute(text(sql), {"version": extractor_version, "providers": providers,
                                            "ids": observation_ids, "limit": limit, "reprocess": reprocess})
            return [Observation(**r._mapping) for r in rows]

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
            })
            if e.mentions:
                conn.execute(_INSERT_MENTION, [{**key, **vars(m)} for m in e.mentions])
            if e.citations:
                conn.execute(_INSERT_CITATION, [{**key, **vars(c)} for c in e.citations])

    def extraction_summary(self, extractor_version: str) -> dict[str, int]:
        sql = """
            SELECT
                (SELECT count(*) FROM observation_extractions WHERE extractor_version = :v) AS extractions,
                (SELECT count(*) FROM observation_extractions
                 WHERE extractor_version = :v AND status = 'failed') AS failed,
                (SELECT count(*) FROM observation_extractions
                 WHERE extractor_version = :v AND validation_status = 'invalid') AS invalid,
                (SELECT count(*) FROM brand_mentions WHERE extractor_version = :v) AS brand_mentions,
                (SELECT count(*) FROM citations WHERE extractor_version = :v) AS citations
        """
        with self._begin() as conn:
            return dict(conn.execute(text(sql), {"v": extractor_version}).one()._mapping)
