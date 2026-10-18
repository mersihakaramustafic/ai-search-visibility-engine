"""Collector orchestration with an in-memory repository and scripted providers."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from visibility_collector.config import RetryConfig
from visibility_collector.providers.base import CollectionTask, ProviderError, ProviderResult
from visibility_collector.repository import DuplicateSuccessError
from visibility_collector.runner import Collector, Selection

FAST_RETRY = RetryConfig(max_attempts=3, initial_backoff_seconds=0.001, max_backoff_seconds=0.001)


def task(request_id: str, provider: str = "openai", query_id: str = "Q001") -> CollectionTask:
    return CollectionTask(request_id=request_id, query_id=query_id, query_text=f"text {query_id}",
                          provider=provider, run_number=1, market="US", language="en", search_enabled=True)


def result(text: str = "answer") -> ProviderResult:
    return ProviderResult(model_id="model-x", response_text=text, request_params={"model": "model-x"},
                          search_config={"enabled": True}, raw_response={"id": "r"},
                          grounding_metadata=None, input_tokens=1, output_tokens=2)


def transient(retry_after=None) -> ProviderError:
    return ProviderError("RateLimitError", "slow down", transient=True, retry_after=retry_after)


class FakeRepo:
    def __init__(self, tasks: list[CollectionTask], succeeded: set[str] = frozenset()):
        self.tasks = {t.request_id: t for t in tasks}
        self.status = {t.request_id: "pending" for t in tasks}
        self.observations: list[dict] = []
        self.claims: list[tuple] = []
        self.lock = threading.Lock()
        for request_id in succeeded:
            self.observations.append({"request_id": request_id, "status": "succeeded", "attempt_number": 1})

    def claim_next(self, provider, statuses, query_ids):
        with self.lock:
            self.claims.append((provider, tuple(statuses), tuple(query_ids) if query_ids else None))
            for rid, t in self.tasks.items():
                if (t.provider == provider and self.status[rid] in statuses
                        and (query_ids is None or t.query_id in query_ids)):
                    self.status[rid] = "in_progress"
                    return t
            return None

    def has_success(self, request_id):
        return any(o["request_id"] == request_id and o["status"] == "succeeded" for o in self.observations)

    def set_request_status(self, request_id, status):
        self.status[request_id] = status

    def record_attempt(self, attempt, request_status):
        with self.lock:
            if attempt.status == "succeeded" and self.has_success(attempt.request_id):
                raise DuplicateSuccessError(attempt.request_id)
            number = 1 + sum(o["request_id"] == attempt.request_id for o in self.observations)
            self.observations.append({**vars(attempt), "attempt_number": number})
            if request_status is not None:
                self.status[attempt.request_id] = request_status
            return number

    def attempts(self, request_id):
        return [o for o in self.observations if o["request_id"] == request_id]


class ScriptedClient:
    """Returns or raises the scripted outcomes in order, per request."""

    def __init__(self, name, script):
        self.name = name
        self.model = "model-x"
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def generate(self, t):
        with self.lock:
            self.calls.append(t.request_id)
            outcome = self.script[t.request_id].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def collector(repo, *clients, retry=FAST_RETRY, concurrency=1):
    return Collector(repo, {c.name: c for c in clients}, retry, {c.name: concurrency for c in clients},
                     now=lambda: datetime(2026, 10, 17, tzinfo=timezone.utc))


def test_success_records_one_observation_and_completes():
    repo = FakeRepo([task("R1")])
    client = ScriptedClient("openai", {"R1": [result("hello")]})
    summary = collector(repo, client).run(7, Selection(statuses=["pending"]))

    assert repo.status["R1"] == "completed"
    [obs] = repo.attempts("R1")
    assert obs["status"] == "succeeded" and obs["response_text"] == "hello"
    assert obs["collection_run_id"] == 7 and obs["attempt_number"] == 1
    assert obs["model_id"] == "model-x" and obs["latency_ms"] >= 0
    assert summary["openai"].succeeded == 1 and summary["openai"].api_calls == 1


def test_transient_error_is_retried_and_every_attempt_is_recorded():
    repo = FakeRepo([task("R1")])
    client = ScriptedClient("openai", {"R1": [transient(), result()]})
    summary = collector(repo, client).run(1, Selection(statuses=["pending"]))

    attempts = repo.attempts("R1")
    assert [(o["attempt_number"], o["status"]) for o in attempts] == [(1, "failed"), (2, "succeeded")]
    assert attempts[0]["error_type"] == "RateLimitError" and attempts[0]["error_message"] == "slow down"
    assert repo.status["R1"] == "completed"
    assert summary["openai"].api_calls == 2


def test_retries_exhausted_marks_request_failed():
    repo = FakeRepo([task("R1")])
    client = ScriptedClient("openai", {"R1": [transient(), transient(), transient()]})
    summary = collector(repo, client).run(1, Selection(statuses=["pending"]))

    assert len(repo.attempts("R1")) == 3
    assert all(o["status"] == "failed" for o in repo.attempts("R1"))
    assert repo.status["R1"] == "failed"
    assert summary["openai"].failed == 1


def test_permanent_error_is_not_retried():
    repo = FakeRepo([task("R1"), task("R2")])
    bad = ProviderError("BadRequestError", "invalid", transient=False)
    client = ScriptedClient("openai", {"R1": [bad], "R2": [result()]})
    collector(repo, client).run(1, Selection(statuses=["pending"]))

    assert len(repo.attempts("R1")) == 1
    assert repo.status == {"R1": "failed", "R2": "completed"}


def test_fatal_error_stops_provider_and_returns_request_to_pending():
    repo = FakeRepo([task("R1"), task("R2"), task("G1", provider="gemini")])
    fatal = ProviderError("AuthenticationError", "bad key", transient=False, fatal=True)
    openai_client = ScriptedClient("openai", {"R1": [fatal], "R2": [result()]})
    gemini_client = ScriptedClient("gemini", {"G1": [result()]})
    summary = collector(repo, openai_client, gemini_client).run(1, Selection(statuses=["pending"]))

    assert repo.status == {"R1": "pending", "R2": "pending", "G1": "completed"}
    assert openai_client.calls == ["R1"]  # R2 never attempted
    assert summary["openai"].stopped_reason and summary["openai"].returned_to_pending == 1
    assert summary["gemini"].succeeded == 1


def test_request_with_existing_success_is_not_called_again():
    repo = FakeRepo([task("R1")], succeeded={"R1"})
    client = ScriptedClient("openai", {"R1": []})
    summary = collector(repo, client).run(1, Selection(statuses=["pending"]))

    assert client.calls == []
    assert repo.status["R1"] == "completed"
    assert summary["openai"].already_collected == 1
    assert len(repo.attempts("R1")) == 1  # no duplicate observation


def test_duplicate_success_from_concurrent_worker_is_discarded():
    repo = FakeRepo([task("R1")])

    class RacingClient(ScriptedClient):
        def generate(self, t):
            out = super().generate(t)
            # Another collector stores a success while this call is in flight.
            repo.observations.append({"request_id": "R1", "status": "succeeded", "attempt_number": 1})
            return out

    summary = collector(repo, RacingClient("openai", {"R1": [result()]})).run(1, Selection(statuses=["pending"]))
    assert len([o for o in repo.attempts("R1") if o["status"] == "succeeded"]) == 1
    assert repo.status["R1"] == "completed"
    assert summary["openai"].already_collected == 1


def test_unexpected_exception_is_recorded_as_failed_attempt():
    repo = FakeRepo([task("R1")])
    client = ScriptedClient("openai", {"R1": [KeyError("candidates")]})
    collector(repo, client).run(1, Selection(statuses=["pending"]))

    [obs] = repo.attempts("R1")
    assert obs["status"] == "failed" and obs["error_type"] == "unexpected:KeyError"
    assert repo.status["R1"] == "failed"


def test_limit_per_provider_holds_across_concurrent_workers():
    tasks = [task(f"R{i}") for i in range(10)]
    repo = FakeRepo(tasks)
    client = ScriptedClient("openai", {t.request_id: [result()] for t in tasks})
    summary = collector(repo, client, concurrency=4).run(1, Selection(statuses=["pending"], limit_per_provider=3))

    assert len(client.calls) == 3
    assert summary["openai"].claimed == 3 and summary["openai"].succeeded == 3
    assert list(repo.status.values()).count("pending") == 7


def test_selection_is_passed_to_claims():
    repo = FakeRepo([task("R1", query_id="Q001"), task("R2", query_id="Q999")])
    client = ScriptedClient("openai", {"R1": [result()], "R2": [result()]})
    collector(repo, client).run(1, Selection(statuses=["pending", "failed"], query_ids=["Q001"]))

    assert client.calls == ["R1"]
    assert repo.claims[0] == ("openai", ("pending", "failed"), ("Q001",))


def test_stop_during_backoff_returns_request_to_pending():
    repo = FakeRepo([task("R1")])
    slow_retry = RetryConfig(max_attempts=3, initial_backoff_seconds=30, max_backoff_seconds=30)
    client = ScriptedClient("openai", {"R1": [transient()]})
    c = collector(repo, client, retry=slow_retry)
    c.stop_event.set()  # as if Ctrl-C arrived while waiting to retry
    outcome, calls = c.process(repo.claim_next("openai", ["pending"], None), 1)

    assert (outcome, calls) == ("interrupted", 1)
    assert repo.status["R1"] == "pending"


@pytest.mark.parametrize("attempt, retry_after, jitter, expected", [
    (1, None, 1.0, 2.0),     # initial
    (3, None, 1.0, 8.0),     # exponential
    (10, None, 1.0, 60.0),   # capped
    (1, None, 0.0, 1.0),     # jitter halves at most
    (1, 20.0, 0.0, 20.0),    # Retry-After wins when longer
    (1, 500.0, 0.0, 60.0),   # but never beyond the cap
])
def test_backoff(attempt, retry_after, jitter, expected):
    c = Collector(FakeRepo([]), {}, RetryConfig(4, 2.0, 60.0), {}, jitter=lambda: jitter)
    assert c.backoff(attempt, retry_after) == pytest.approx(expected)
