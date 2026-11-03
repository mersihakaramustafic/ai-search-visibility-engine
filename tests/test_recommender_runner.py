"""Retries, validation and outcomes per opportunity, with a scripted fake generator (no API calls)."""

from __future__ import annotations

import json

import pytest

from recommender_fixtures import BRAND_DOMAINS, pack
from test_recommender_output import rec
from visibility_collector.providers.base import ProviderError
from visibility_extractor.llm import LLMCall
from visibility_recommender.config import load_config
from visibility_recommender.runner import RecommendationRunner

CONFIG = load_config()
PACK = pack()


class FakeGenerator:
    """Returns the scripted items in order: an output dict, a raw string, or an exception to raise."""

    provider, model = "gemini", "gemini-3.8-flash"

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def generate(self, pack):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMCall(text, "gemini-3.8-flash", 1000, 0, 400, 100, 2000)


def good(**changes):
    return {"opportunity_id": PACK.opportunity_id, "insufficient_evidence": False, "recommendations": [rec()],
            **changes}


def run(*script, p=PACK):
    sleeps = []
    gen = FakeGenerator(*script)
    runner = RecommendationRunner(gen, CONFIG, BRAND_DOMAINS, sleep=sleeps.append, jitter=lambda: 1.0)
    return runner.process(p), gen, sleeps


def transient(retry_after=None):
    return ProviderError("ClientError:429:RESOURCE_EXHAUSTED", "rate limited", transient=True, retry_after=retry_after)


def test_success_records_provenance_usage_and_cost():
    g, gen, _ = run(good())
    assert (g.status, g.attempts, gen.calls) == ("succeeded", 1, 1)
    assert (g.provider, g.model_requested, g.model_id, g.prompt_version, g.schema_version, g.generator_version) == (
        "gemini", "gemini-3.8-flash", "gemini-3.8-flash", "r1", "s1", "recs-v2")
    assert g.output.recommendations[0].title == rec()["title"]
    assert g.raw_output == good()
    assert (g.input_tokens, g.output_tokens, g.latency_ms) == (1000, 400, 2000)
    assert g.estimated_cost_usd == pytest.approx((1000 * 0.75 + 400 * 3.75) / 1e6)


def test_insufficient_pack_is_not_sent():
    p = pack(gap_type="weaker_on_provider", competitors=(), competitor_value=None, supporting_domains=(),
             compared_provider="gemini", compared_value=0.8)
    g, gen, _ = run(p=p)
    assert (g.status, g.attempts, gen.calls, g.estimated_cost_usd) == ("insufficient_evidence", 0, 0, None)
    assert "nothing specific" in g.insufficient_reason


def test_model_can_report_insufficient_evidence():
    g, _, _ = run(good(recommendations=[], insufficient_evidence=True, insufficient_reason="Only rankings, no sources."))
    assert (g.status, g.insufficient_reason, g.output.recommendations) == (
        "insufficient_evidence", "Only rankings, no sources.", [])


def test_invalid_output_is_retried_then_accepted():
    fabricated = good(recommendations=[rec(evidence_ids=["metric:hubspot", "domain:capterra.com"])])
    g, gen, sleeps = run("not json", fabricated, good())
    assert (g.status, g.attempts, gen.calls, sleeps) == ("succeeded", 3, 3, [])
    assert (g.input_tokens, g.error_type, g.validation_errors) == (3000, None, [])
    assert g.estimated_cost_usd == pytest.approx(3 * (1000 * 0.75 + 400 * 3.75) / 1e6)  # every attempt is paid


def test_invalid_on_every_attempt_fails_and_keeps_the_last_output():
    overclaim = good(recommendations=[rec(rationale="This guarantees more citations for HubSpot in OpenAI answers.")])
    g, gen, _ = run("not json", "still not json", overclaim)
    assert (g.status, g.attempts, g.error_type, g.output) == ("failed", 3, "invalid_output", None)
    assert g.raw_output == overclaim
    assert any("certain outcome" in e for e in g.validation_errors)


def test_transient_errors_back_off_and_honour_retry_after():
    g, gen, sleeps = run(transient(), transient(retry_after=30), good())
    assert (g.status, g.attempts) == ("succeeded", 3)
    assert sleeps == [2.0, 30.0]


def test_transient_errors_on_every_attempt_fail():
    g, _, sleeps = run(transient(), transient(), transient())
    assert (g.status, g.error_type, g.attempts, len(sleeps)) == ("failed", "ClientError:429:RESOURCE_EXHAUSTED", 3, 2)


def test_permanent_error_fails_without_retry():
    g, gen, _ = run(ProviderError("incomplete_response", "finish_reason=SAFETY", transient=False))
    assert (g.status, g.error_type, gen.calls) == ("failed", "incomplete_response", 1)


def test_fatal_error_stops_the_run():
    with pytest.raises(ProviderError):
        run(ProviderError("ClientError:404:NOT_FOUND", "model not found", transient=False, fatal=True))
