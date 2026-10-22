"""Rules + LLM orchestration with a scripted fake classifier (no API calls, no database)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from brand_fixtures import BRANDS
from visibility_collector.config import RetryConfig
from visibility_collector.providers.base import ProviderError
from visibility_extractor.config import load_config
from visibility_extractor.llm import LLMCall, load_schema
from visibility_extractor.pipeline import Observation, RulesExtractor
from visibility_extractor.runner import ExtractionRunner, backoff

CONFIG = replace(load_config(), retry=RetryConfig(max_attempts=3, initial_backoff_seconds=2, max_backoff_seconds=60))
ANSWER = ("For small teams, HubSpot is the best pick. Pipedrive is cheaper, but its reporting is weak. "
          "Close works well if you make many calls. Close deals faster with any of them.")


class FakeClassifier:
    """Returns the scripted items in order: an output dict, a raw string, or an exception to raise."""

    def __init__(self, *script):
        self.script = list(script)
        self.schema = load_schema()
        self.requests = []

    def classify(self, request):
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMCall(text, "gpt-5.6-luna", 1000, 800, 200, 50, 900)


def output(close_is_brand=(True, False), hubspot="primary", sentiment="positive"):
    return {
        "brands": [
            {"brand_id": "hubspot", "sentiment": sentiment, "recommendation": hubspot,
             "evidence": "HubSpot is the best pick"},
            {"brand_id": "pipedrive", "sentiment": "negative", "recommendation": "none",
             "evidence": "its reporting is weak"},
            {"brand_id": "close", "sentiment": "positive", "recommendation": "conditional",
             "evidence": "Close works well if you make many calls"},
        ],
        "candidates": [
            {"candidate_id": 1, "is_brand": close_is_brand[0], "evidence": "Close works well"},
            {"candidate_id": 2, "is_brand": close_is_brand[1], "evidence": "Close deals faster"},
        ],
    }


def run(classifier, text=ANSWER, provider="openai", config=CONFIG):
    sleeps = []
    runner = ExtractionRunner(RulesExtractor(BRANDS, config), config, classifier,
                              sleep=sleeps.append, jitter=lambda: 1.0)
    return runner.process(Observation(5, provider, text, None, "Best CRM for a small team?")), sleeps


def transient(retry_after=None):
    return ProviderError("RateLimitError", "slow down", transient=True, retry_after=retry_after)


# --- success ----------------------------------------------------------------------------------


def test_labels_are_applied_and_accepted_candidates_become_mentions():
    e, _ = run(FakeClassifier(output()))
    assert (e.status, e.validation_status, e.strategy, e.llm.status) == ("succeeded", "valid", "rules_and_llm",
                                                                         "succeeded")
    assert [(m.brand_id, m.mention_count, m.match_method, m.sentiment, m.recommendation) for m in e.mentions] == [
        ("hubspot", 1, "alias", "positive", "primary"),
        ("pipedrive", 1, "alias", "negative", "none"),
        ("close", 1, "llm_disambiguated", "positive", "conditional"),  # only candidate 1 accepted
    ]
    assert e.mentions[2].classification_evidence == "Close works well if you make many calls"
    assert (e.target_brand_mentioned, e.target_brand_recommended, e.competitor_brands_mentioned) == (True, True, 2)
    assert e.candidate_count == 2


def test_rejected_candidates_add_nothing():
    e, _ = run(FakeClassifier(output(close_is_brand=(False, False))))
    assert [m.brand_id for m in e.mentions] == ["hubspot", "pipedrive"]
    assert e.competitor_brands_mentioned == 1


@pytest.mark.parametrize("recommendation, recommended", [("primary", True), ("conditional", True), ("none", False)])
def test_target_recommended_only_when_the_label_says_so(recommendation, recommended):
    e, _ = run(FakeClassifier(output(hubspot=recommendation)))
    assert e.target_brand_mentioned and e.target_brand_recommended is recommended


@pytest.mark.parametrize("sentiment", ["positive", "neutral", "negative"])
def test_sentiment_is_stored_per_brand(sentiment):
    e, _ = run(FakeClassifier(output(sentiment=sentiment)))
    assert e.mentions[0].sentiment == sentiment


def test_provenance_and_usage_are_recorded():
    e, _ = run(FakeClassifier(output()))
    r = e.llm
    assert (r.model_requested, r.model_id, r.prompt_version, r.schema_version) == (
        "gpt-5.6-luna", "gpt-5.6-luna", CONFIG.llm.prompt_version, CONFIG.llm.schema_version)
    assert (r.attempts, r.input_tokens, r.cached_input_tokens, r.output_tokens, r.reasoning_tokens) == (
        1, 1000, 800, 200, 50)
    assert r.estimated_cost_usd == pytest.approx((200 * 0.20 + 800 * 0.02 + 200 * 1.20) / 1e6)
    assert [c["is_brand"] for c in r.output["candidates"]] == [True, False]


def test_query_text_is_sent_with_the_answer():
    fake = FakeClassifier(output())
    run(fake)
    assert fake.requests[0].query_text == "Best CRM for a small team?"


# --- no call ------------------------------------------------------------------------------------


def test_nothing_to_classify_makes_no_call():
    fake = FakeClassifier()
    e, _ = run(fake, text="A CRM stores customer data.")
    assert fake.requests == [] and e.llm.status == "not_needed" and e.llm.attempts == 0
    assert (e.target_brand_mentioned, e.target_brand_recommended) == (False, False)


def test_rules_only_provider_makes_no_call():
    config = replace(CONFIG, strategy={"openai": "rules_and_llm"})
    e, _ = run(FakeClassifier(), provider="gemini", config=config)
    assert (e.strategy, e.llm, e.target_brand_recommended) == ("rules_only", None, None)
    assert e.mentions[0].sentiment is None


def test_failed_rules_step_does_not_reach_the_llm():
    fake = FakeClassifier()
    e, _ = run(fake, text=None)
    assert (e.status, e.llm, e.strategy) == ("failed", None, "rules_and_llm") and fake.requests == []


# --- retries --------------------------------------------------------------------------------------


def test_transient_errors_are_retried_with_backoff():
    e, sleeps = run(FakeClassifier(transient(), transient(retry_after=30), output()))
    assert e.llm.status == "succeeded" and e.llm.attempts == 3
    assert sleeps == [2.0, 30.0]  # 2s backoff, then the server's Retry-After wins over 4s


def test_invalid_output_is_retried_and_its_tokens_are_counted():
    bad = output()
    bad["brands"][0]["evidence"] = "HubSpot is the only CRM anyone needs"
    e, sleeps = run(FakeClassifier(bad, output()))
    assert e.llm.status == "succeeded" and e.llm.attempts == 2 and sleeps == []
    assert (e.llm.input_tokens, e.llm.output_tokens) == (2000, 400)


def test_all_attempts_failing_marks_the_extraction_failed():
    e, sleeps = run(FakeClassifier(transient(), transient(), transient()))
    assert (e.status, e.validation_status, e.error_type, e.llm.status, e.llm.attempts) == (
        "failed", None, "RateLimitError", "failed", 3)
    assert e.mentions == [] and e.citations == [] and len(sleeps) == 2


def test_invalid_output_every_time_keeps_reasons_and_cost():
    e, _ = run(FakeClassifier("not json", "{}", '{"brands": [], "candidates": []}'))
    assert (e.status, e.error_type, e.llm.attempts, e.llm.input_tokens) == ("failed", "invalid_output", 3, 3000)
    assert e.validation_issues and e.llm.estimated_cost_usd > 0


def test_non_transient_error_is_not_retried():
    e, sleeps = run(FakeClassifier(ProviderError("incomplete_response", "max_output_tokens", transient=False)))
    assert (e.status, e.error_type, e.llm.attempts, sleeps) == ("failed", "incomplete_response", 1, [])


def test_fatal_error_stops_the_run():
    with pytest.raises(ProviderError):
        run(FakeClassifier(ProviderError("NotFoundError:model_not_found", "no model", transient=False, fatal=True)))


def test_backoff_doubles_and_is_capped():
    retry = RetryConfig(max_attempts=9, initial_backoff_seconds=2, max_backoff_seconds=60)
    assert [backoff(retry, a, None, 1.0) for a in (1, 2, 3, 7)] == [2, 4, 8, 60]
    assert backoff(retry, 1, None, 0.0) == 1.0  # jitter halves at most
    assert backoff(retry, 1, 500, 1.0) == 60  # Retry-After is capped too


def test_output_is_kept_on_success_and_on_failure():
    ok, _ = run(FakeClassifier(output()))
    assert ok.llm.output == output()
    failed, _ = run(FakeClassifier(transient(), "not json", transient()))
    assert failed.llm.output == "not json"  # last rejected output, raw because it was not JSON
