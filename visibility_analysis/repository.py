"""Read-only database access for the visibility analysis: brands and per-answer facts."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from visibility_analysis.metrics import Answer

# One row per answer whose extraction by this version succeeded. Citations noted with
# brand_attribution sit on a tracked brand's shared domain, so they are not outside domains.
_ANSWERS = text("""
    SELECT o.observation_id, r.query_id, o.provider, q.intent,
           ARRAY(SELECT m.brand_id FROM brand_mentions m
                 WHERE m.observation_id = e.observation_id AND m.extractor_version = e.extractor_version)
               AS mentioned,
           ARRAY(SELECT m.brand_id FROM brand_mentions m
                 WHERE m.observation_id = e.observation_id AND m.extractor_version = e.extractor_version
                   AND m.recommendation <> 'none') AS recommended,
           ARRAY(SELECT DISTINCT c.brand_id FROM citations c
                 WHERE c.observation_id = e.observation_id AND c.extractor_version = e.extractor_version
                   AND c.brand_id IS NOT NULL) AS cited,
           ARRAY(SELECT DISTINCT c.domain FROM citations c
                 WHERE c.observation_id = e.observation_id AND c.extractor_version = e.extractor_version
                   AND c.brand_id IS NULL AND c.domain IS NOT NULL
                   AND NOT c.metadata ? 'brand_attribution') AS domains
    FROM observation_extractions e
    JOIN observations o ON o.observation_id = e.observation_id
    JOIN collection_requests r ON r.request_id = o.request_id
    JOIN queries q ON q.query_id = r.query_id
    WHERE e.extractor_version = :version AND e.status = 'succeeded' AND o.status = 'succeeded'
      AND (CAST(:providers AS text[]) IS NULL OR o.provider = ANY(CAST(:providers AS text[])))
    ORDER BY o.observation_id
""")


class AnalysisRepository:
    def __init__(self, engine: Engine, begin: Callable[[], AbstractContextManager[Connection]] | None = None):
        self._engine = engine
        self._begin = begin or engine.begin

    def load_brands(self) -> tuple[list[str], str]:
        """All brand ids and the target brand (the one primary brand)."""
        with self._begin() as conn:
            rows = conn.execute(text("SELECT brand_id, is_primary_brand FROM brands ORDER BY brand_id")).all()
        primary = [r.brand_id for r in rows if r.is_primary_brand]
        if len(primary) != 1:
            raise ValueError(f"expected exactly one primary brand, found {primary}")
        return [r.brand_id for r in rows], primary[0]

    def load_answers(self, extractor_version: str, providers: list[str] | None = None) -> list[Answer]:
        with self._begin() as conn:
            rows = conn.execute(_ANSWERS, {"version": extractor_version, "providers": providers})
            return [Answer(r.observation_id, r.query_id, r.provider, r.intent, frozenset(r.mentioned),
                           frozenset(r.recommended), frozenset(r.cited), frozenset(r.domains)) for r in rows]
