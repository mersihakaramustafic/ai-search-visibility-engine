"""Database access for the collector: claim requests, record attempts, track runs.

Safety properties:
- A request is claimed with FOR UPDATE SKIP LOCKED and moved to 'in_progress',
  so parallel workers (or two collector processes) never take the same one.
- An observation and the request's new status are written in one transaction.
- The partial unique index from migration 0002 allows at most one succeeded
  observation per request, so a retry can never duplicate a success.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from visibility_collector.providers.base import CollectionTask

ONE_SUCCESS_INDEX = "observations_one_success_per_request"


@dataclass(frozen=True)
class Attempt:
    """One API call, ready to be stored as an observations row."""

    request_id: str
    collection_run_id: int
    provider: str
    status: str  # 'succeeded' | 'failed'
    model_id: str | None
    request_params: dict[str, Any]
    search_config: dict[str, Any]
    response_text: str | None
    raw_response: dict[str, Any] | None
    grounding_metadata: dict[str, Any] | None
    error_type: str | None
    error_message: str | None
    latency_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    requested_at: datetime
    responded_at: datetime | None


class DuplicateSuccessError(Exception):
    """The request already has a succeeded observation."""


_CLAIM_COLUMNS = """cr.request_id, cr.query_id, q.query_text, cr.provider, cr.run_number,
                    cr.market, cr.language, cr.search_enabled"""

_SELECTION = """cr.provider = :provider
                AND cr.status = ANY(:statuses)
                AND (CAST(:query_ids AS text[]) IS NULL OR cr.query_id = ANY(CAST(:query_ids AS text[])))"""


def _task(row: Any) -> CollectionTask:
    return CollectionTask(**row._mapping)


class Repository:
    def __init__(self, engine: Engine, begin: Callable[[], AbstractContextManager[Connection]] | None = None):
        self._engine = engine
        # Overridable so a smoke test can run everything inside one rolled-back transaction.
        self._begin = begin or engine.begin

    # --- lookups -----------------------------------------------------------

    def existing_query_ids(self, query_ids: list[str]) -> set[str]:
        with self._begin() as conn:
            rows = conn.execute(text("SELECT query_id FROM queries WHERE query_id = ANY(:ids)"), {"ids": query_ids})
            return set(rows.scalars())

    def preview(self, provider: str, statuses: list[str], query_ids: list[str] | None,
                limit: int | None) -> list[CollectionTask]:
        """Requests a run would claim, without claiming them (for --dry-run)."""
        sql = f"""
            SELECT {_CLAIM_COLUMNS}
            FROM collection_requests cr JOIN queries q ON q.query_id = cr.query_id
            WHERE {_SELECTION}
            ORDER BY cr.query_id, cr.run_number
            LIMIT :limit
        """
        with self._begin() as conn:
            rows = conn.execute(text(sql), {"provider": provider, "statuses": statuses,
                                            "query_ids": query_ids, "limit": limit})
            return [_task(r) for r in rows]

    def status_counts(self, provider: str, query_ids: list[str] | None) -> dict[str, int]:
        sql = """
            SELECT status, count(*) FROM collection_requests cr
            WHERE cr.provider = :provider
              AND (CAST(:query_ids AS text[]) IS NULL OR cr.query_id = ANY(CAST(:query_ids AS text[])))
            GROUP BY status
        """
        with self._begin() as conn:
            return dict(conn.execute(text(sql), {"provider": provider, "query_ids": query_ids}).all())

    # --- runs ----------------------------------------------------------------

    def create_run(self, run_label: str, collector_config: dict[str, Any], code_version: str | None) -> int:
        sql = text("""
            INSERT INTO collection_runs (run_label, status, collector_config, code_version, started_at)
            VALUES (:run_label, 'running', :collector_config, :code_version, now())
            RETURNING collection_run_id
        """).bindparams(bindparam("collector_config", type_=JSONB))
        with self._begin() as conn:
            return conn.execute(sql, {"run_label": run_label, "collector_config": collector_config,
                                      "code_version": code_version}).scalar_one()

    def finish_run(self, run_id: int, status: str, notes: str) -> None:
        with self._begin() as conn:
            conn.execute(
                text("UPDATE collection_runs SET status = :status, notes = :notes, finished_at = now() "
                     "WHERE collection_run_id = :run_id"),
                {"status": status, "notes": notes, "run_id": run_id},
            )

    # --- requests --------------------------------------------------------------

    def requeue_in_progress(self, provider: str, query_ids: list[str] | None) -> int:
        """Return requests stuck in 'in_progress' (e.g. after a crash) to 'pending'."""
        sql = f"""
            UPDATE collection_requests cr SET status = 'pending'
            WHERE {_SELECTION}
        """
        with self._begin() as conn:
            result = conn.execute(text(sql), {"provider": provider, "statuses": ["in_progress"],
                                              "query_ids": query_ids})
            return result.rowcount

    def claim_next(self, provider: str, statuses: list[str], query_ids: list[str] | None) -> CollectionTask | None:
        sql = f"""
            WITH next AS (
                SELECT cr.request_id
                FROM collection_requests cr
                WHERE {_SELECTION}
                ORDER BY cr.query_id, cr.run_number
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE collection_requests cr SET status = 'in_progress'
            FROM next, queries q
            WHERE cr.request_id = next.request_id AND q.query_id = cr.query_id
            RETURNING {_CLAIM_COLUMNS}
        """
        with self._begin() as conn:
            row = conn.execute(text(sql), {"provider": provider, "statuses": statuses,
                                           "query_ids": query_ids}).first()
            return _task(row) if row else None

    def has_success(self, request_id: str) -> bool:
        with self._begin() as conn:
            return conn.execute(
                text("SELECT EXISTS (SELECT 1 FROM observations WHERE request_id = :r AND status = 'succeeded')"),
                {"r": request_id},
            ).scalar_one()

    def set_request_status(self, request_id: str, status: str) -> None:
        with self._begin() as conn:
            conn.execute(text("UPDATE collection_requests SET status = :s WHERE request_id = :r"),
                         {"s": status, "r": request_id})

    def record_attempt(self, attempt: Attempt, request_status: str | None) -> int:
        """Insert the observation and, if given, set the request status, atomically.

        Returns the attempt_number used. Raises DuplicateSuccessError if the
        request already has a succeeded observation.
        """
        insert = text("""
            INSERT INTO observations (
                request_id, collection_run_id, attempt_number, provider, model_id, request_params,
                search_config, status, response_text, raw_response, grounding_metadata, error_type,
                error_message, latency_ms, input_tokens, output_tokens, requested_at, responded_at)
            VALUES (
                :request_id, :collection_run_id,
                (SELECT coalesce(max(attempt_number), 0) + 1 FROM observations WHERE request_id = :request_id),
                :provider, :model_id, :request_params, :search_config, :status, :response_text,
                :raw_response, :grounding_metadata, :error_type, :error_message, :latency_ms,
                :input_tokens, :output_tokens, :requested_at, :responded_at)
            RETURNING attempt_number
        """).bindparams(*(bindparam(c, type_=JSONB(none_as_null=True)) for c in
                          ("request_params", "search_config", "raw_response", "grounding_metadata")))
        values = vars(attempt).copy()
        try:
            with self._begin() as conn:
                attempt_number = conn.execute(insert, values).scalar_one()
                if request_status is not None:
                    conn.execute(text("UPDATE collection_requests SET status = :s WHERE request_id = :r"),
                                 {"s": request_status, "r": attempt.request_id})
                return attempt_number
        except IntegrityError as e:
            if ONE_SUCCESS_INDEX in str(e.orig):
                raise DuplicateSuccessError(attempt.request_id) from None
            raise
