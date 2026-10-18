"""Collection orchestration: claim requests, call providers, retry, record.

Each provider gets its own pool of `concurrency` worker threads. A worker
claims one request at a time and makes up to retry.max_attempts API calls
for it. Every call is stored as an observation (failed attempts included),
and the request ends as:

    completed    a succeeded observation was stored (or already existed)
    failed       retries ran out, or the error was not retryable
    pending      the provider hit a fatal error (bad key, no quota, unknown
                 model) or the run was interrupted: the request was never
                 really tried, so a later run picks it up again
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from visibility_collector.config import RetryConfig
from visibility_collector.logs import log
from visibility_collector.providers.base import CollectionTask, ProviderClient, ProviderError, ProviderResult
from visibility_collector.repository import Attempt, DuplicateSuccessError


@dataclass(frozen=True)
class Selection:
    """Which requests a run collects."""

    statuses: list[str]
    query_ids: list[str] | None = None
    limit_per_provider: int | None = None


@dataclass
class ProviderSummary:
    claimed: int = 0
    succeeded: int = 0
    failed: int = 0
    already_collected: int = 0
    returned_to_pending: int = 0
    api_calls: int = 0
    stopped_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class _ProviderState:
    summary: ProviderSummary = field(default_factory=ProviderSummary)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop: threading.Event = field(default_factory=threading.Event)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Collector:
    def __init__(
        self,
        repo: Any,
        clients: dict[str, ProviderClient],
        retry: RetryConfig,
        concurrency: dict[str, int],
        *,
        now: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ):
        self.repo = repo
        self.clients = clients
        self.retry = retry
        self.concurrency = concurrency
        self._now = now
        self._monotonic = monotonic
        self._jitter = jitter
        self.stop_event = threading.Event()

    # --- run ---------------------------------------------------------------

    def run(self, run_id: int, selection: Selection) -> dict[str, ProviderSummary]:
        states = {name: _ProviderState() for name in self.clients}
        workers = sum(self.concurrency.get(name, 1) for name in self.clients)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="collector") as pool:
            futures = [
                pool.submit(self._worker, name, run_id, selection, states[name])
                for name in self.clients
                for _ in range(self.concurrency.get(name, 1))
            ]
            try:
                done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            except KeyboardInterrupt:
                log(logging.WARNING, "interrupted; finishing in-flight API calls before stopping")
                self.stop_event.set()
                wait(futures)
                raise
            for future in done:
                if future.exception() is not None:
                    self.stop_event.set()
                    wait(futures)
                    raise future.exception()
        return {name: state.summary for name, state in states.items()}

    def _worker(self, provider: str, run_id: int, selection: Selection, state: _ProviderState) -> None:
        summary = state.summary
        while not self.stop_event.is_set() and not state.stop.is_set():
            with state.lock:
                if selection.limit_per_provider is not None and summary.claimed >= selection.limit_per_provider:
                    return
                summary.claimed += 1  # reserve a slot before claiming so the limit holds across threads
            task = self.repo.claim_next(provider, selection.statuses, selection.query_ids)
            if task is None:
                with state.lock:
                    summary.claimed -= 1
                return
            outcome, calls = self.process(task, run_id)
            with state.lock:
                summary.api_calls += calls
                if outcome == "succeeded":
                    summary.succeeded += 1
                elif outcome == "failed":
                    summary.failed += 1
                elif outcome == "already_collected":
                    summary.already_collected += 1
                else:  # fatal or interrupted
                    summary.returned_to_pending += 1
                if outcome == "fatal" and not state.stop.is_set():
                    summary.stopped_reason = "fatal provider error; see logs"
                    state.stop.set()

    # --- one request ---------------------------------------------------------

    def process(self, task: CollectionTask, run_id: int) -> tuple[str, int]:
        """Collect one claimed request. Returns (outcome, number of API calls)."""
        fields = {"request_id": task.request_id, "provider": task.provider}
        if self.repo.has_success(task.request_id):
            self.repo.set_request_status(task.request_id, "completed")
            log(logging.INFO, "already collected; marked completed", **fields)
            return "already_collected", 0

        client = self.clients[task.provider]
        for attempt in range(1, self.retry.max_attempts + 1):
            requested_at = self._now()
            started = self._monotonic()
            try:
                result = client.generate(task)
            except ProviderError as e:
                outcome, delay = self._handle_error(task, run_id, e, attempt, requested_at, started)
                if outcome is not None:
                    return outcome, attempt
                # Waits on the stop event so Ctrl-C does not sit out a long backoff.
                if self.stop_event.wait(delay):
                    self.repo.set_request_status(task.request_id, "pending")
                    return "interrupted", attempt
                continue
            except Exception as e:  # SDK or parsing bug: record it, don't retry blindly
                error = ProviderError(f"unexpected:{type(e).__name__}", str(e), transient=False)
                outcome, _ = self._handle_error(task, run_id, error, attempt, requested_at, started)
                return outcome or "failed", attempt
            return self._record_success(task, run_id, result, requested_at, started), attempt
        raise AssertionError("unreachable: the last attempt always returns")

    def _record_success(self, task: CollectionTask, run_id: int, result: ProviderResult,
                        requested_at: datetime, started: float) -> str:
        latency_ms = int((self._monotonic() - started) * 1000)
        attempt = Attempt(
            request_id=task.request_id, collection_run_id=run_id, provider=task.provider, status="succeeded",
            model_id=result.model_id, request_params=result.request_params, search_config=result.search_config,
            response_text=result.response_text, raw_response=result.raw_response,
            grounding_metadata=result.grounding_metadata, error_type=None, error_message=None,
            latency_ms=latency_ms, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            requested_at=requested_at, responded_at=self._now(),
        )
        try:
            attempt_number = self.repo.record_attempt(attempt, "completed")
        except DuplicateSuccessError:
            self.repo.set_request_status(task.request_id, "completed")
            log(logging.WARNING, "success already recorded by another worker; discarded duplicate",
                request_id=task.request_id, provider=task.provider)
            return "already_collected"
        log(logging.INFO, "succeeded", request_id=task.request_id, provider=task.provider,
            attempt=attempt_number, model_id=result.model_id, latency_ms=latency_ms,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            grounded=result.grounding_metadata is not None)
        return "succeeded"

    def _handle_error(self, task: CollectionTask, run_id: int, e: ProviderError, attempt: int,
                      requested_at: datetime, started: float) -> tuple[str | None, float]:
        """Store the failed attempt. Returns (final outcome or None to retry, backoff delay)."""
        latency_ms = int((self._monotonic() - started) * 1000)
        retry = e.transient and not e.fatal and attempt < self.retry.max_attempts
        if retry:
            new_status, outcome = None, None  # stays in_progress while retrying
        elif e.fatal:
            new_status, outcome = "pending", "fatal"
        else:
            new_status, outcome = "failed", "failed"
        record = Attempt(
            request_id=task.request_id, collection_run_id=run_id, provider=task.provider, status="failed",
            model_id=e.model_id, request_params=e.request_params, search_config=e.search_config,
            response_text=None, raw_response=e.raw_response, grounding_metadata=None,
            error_type=e.error_type, error_message=e.message, latency_ms=latency_ms,
            input_tokens=None, output_tokens=None, requested_at=requested_at, responded_at=self._now(),
        )
        attempt_number = self.repo.record_attempt(record, new_status)
        delay = self.backoff(attempt, e.retry_after) if retry else 0.0
        log(logging.WARNING if retry else logging.ERROR, "attempt failed",
            request_id=task.request_id, provider=task.provider, attempt=attempt_number,
            error_type=e.error_type, transient=e.transient, fatal=e.fatal, latency_ms=latency_ms,
            next=("retry" if retry else new_status), retry_in_s=round(delay, 1) if retry else None)
        return outcome, delay

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with jitter, never shorter than the server's Retry-After."""
        base = min(self.retry.max_backoff_seconds, self.retry.initial_backoff_seconds * 2 ** (attempt - 1))
        delay = base * (0.5 + self._jitter() / 2)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.retry.max_backoff_seconds))
        return delay
