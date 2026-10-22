"""LLM request building, API handling, output validation and cost, with a fake OpenAI client (no API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from brand_fixtures import BRANDS
from visibility_collector.providers.base import ProviderError
from visibility_extractor.config import ExtractionConfigError, Price, load_config, parse_config
from visibility_extractor.llm import (
    ClassificationRequest,
    LLMClassifier,
    build_request,
    estimate_cost,
    load_schema,
    validate_output,
)
from visibility_extractor.matching import BrandMatcher, aggregate_mentions

CONFIG = load_config()
SCHEMA = load_schema()
ANSWER = ("For small teams, HubSpot is the best pick. Pipedrive is a cheaper option, but its reporting is weak. "
          "Close works well if you make many calls. Close deals faster with any of them.")


def request() -> ClassificationRequest:
    matcher = BrandMatcher(BRANDS, CONFIG.matching)
    match = matcher.match(ANSWER)
    mentions = aggregate_mentions(ANSWER, match.confirmed, 20)
    return build_request(7, "Best CRM for a small team?", ANSWER, BRANDS, mentions, match.candidates, 20)


def good_output(**changes) -> dict:
    out = {
        "brands": [
            {"brand_id": "hubspot", "sentiment": "positive", "recommendation": "primary",
             "evidence": "HubSpot is the best pick"},
            {"brand_id": "pipedrive", "sentiment": "negative", "recommendation": "conditional",
             "evidence": "Pipedrive is a cheaper option, but its reporting is weak"},
            {"brand_id": "close", "sentiment": "neutral", "recommendation": "conditional",
             "evidence": "Close works well if you make many calls"},
        ],
        "candidates": [
            {"candidate_id": 1, "is_brand": True, "evidence": "Close works well"},
            {"candidate_id": 2, "is_brand": False, "evidence": "Close deals faster"},
        ],
    }
    out.update(changes)
    return out


# --- request ------------------------------------------------------------------------------


def test_request_has_mentioned_and_candidate_brands_with_numbered_candidates():
    r = request()
    assert [b["brand_id"] for b in r.brands] == ["hubspot", "pipedrive", "close"]
    assert r.brands[2]["aliases"] == ["Close", "Close CRM", "Close.com", "Close.io"]
    assert [(c["candidate_id"], c["text"], ANSWER[c["char_offset"]:c["char_offset"] + 5]) for c in r.candidates] == [
        (1, "Close", "Close"), (2, "Close", "Close")]
    payload = json.loads(r.payload())
    assert set(payload) == {"question", "answer", "brands", "candidates"}


def test_nothing_to_classify_means_no_request():
    assert build_request(1, "q", "A CRM stores data.", BRANDS, [], [], 20) is None


# --- API call -------------------------------------------------------------------------------


def fake_response(output_text, status="completed", reason=None):
    return SimpleNamespace(
        status=status, output_text=output_text, model="gpt-5.6-luna-2026-07-01",
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300,
                              input_tokens_details=SimpleNamespace(cached_tokens=1000),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=120)))


class FakeClient:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result, self.error = result, error
        self.responses = SimpleNamespace(create=self._create)
        self.models = SimpleNamespace(retrieve=self._retrieve)

    def _create(self, **params):
        self.calls.append(params)
        if self.error:
            raise self.error
        return self.result

    def _retrieve(self, model):
        if self.error:
            raise self.error
        return SimpleNamespace(id=model)


def classifier(client):
    return LLMClassifier(client, CONFIG.llm, SCHEMA)


def api_error(cls, status, headers=None, body=None):
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return cls("boom", response=httpx.Response(status, headers=headers or {}, request=req), body=body)


def test_request_uses_configured_model_strict_schema_and_low_effort():
    client = FakeClient(fake_response(json.dumps(good_output())))
    classifier(client).classify(request())
    [params] = client.calls
    assert params["model"] == "gpt-5.6-luna"
    assert params["text"]["format"] == {"type": "json_schema", "name": "brand_classification",
                                        "schema": SCHEMA, "strict": True}
    assert params["reasoning"] == {"effort": "low"} and params["store"] is False
    assert "Use only the answer text" in params["instructions"]


def test_usage_is_read_including_cached_and_reasoning_tokens():
    call = classifier(FakeClient(fake_response("{}"))).classify(request())
    assert (call.model_id, call.input_tokens, call.cached_input_tokens, call.output_tokens, call.reasoning_tokens) == (
        "gpt-5.6-luna-2026-07-01", 1200, 1000, 300, 120)


def test_model_can_be_overridden_from_the_environment():
    raw = {"extractor_version": "v1", "matching": {"max_candidates_per_brand": 1, "snippet_chars": 1},
           "llm": {"model": "gpt-5.6-luna", "api_key_env": "OPENAI_API_KEY", "timeout_seconds": 5,
                   "max_output_tokens": 100, "reasoning_effort": "low", "prompt_version": "p1",
                   "schema_version": "s1"}}
    assert parse_config(raw).llm.model == "gpt-5.6-luna"
    assert parse_config(raw, {"EXTRACTION_MODEL": "gpt-other"}).llm.model == "gpt-other"
    with pytest.raises(ExtractionConfigError, match="reasoning_effort"):
        parse_config({**raw, "llm": {**raw["llm"], "reasoning_effort": "max"}})


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.setattr("visibility_db.config.load_dotenv_file", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ExtractionConfigError, match="OPENAI_API_KEY is not set"):
        LLMClassifier.from_config(CONFIG.llm)


@pytest.mark.parametrize("error, transient, fatal, retry_after", [
    (openai.APITimeoutError(request=httpx.Request("POST", "https://x")), True, False, None),
    (api_error(openai.RateLimitError, 429, {"retry-after": "7"}), True, False, 7.0),
    (api_error(openai.InternalServerError, 503), True, False, None),
    (api_error(openai.NotFoundError, 404, body={"code": "model_not_found"}), False, True, None),
    (api_error(openai.AuthenticationError, 401), False, True, None),
    (api_error(openai.BadRequestError, 400), False, False, None),
])
def test_api_errors_are_classified_for_retries(error, transient, fatal, retry_after):
    with pytest.raises(ProviderError) as exc:
        classifier(FakeClient(error=error)).classify(request())
    assert (exc.value.transient, exc.value.fatal, exc.value.retry_after) == (transient, fatal, retry_after)


def test_incomplete_response_is_an_error():
    with pytest.raises(ProviderError, match="max_output_tokens") as exc:
        classifier(FakeClient(fake_response("{", "incomplete", "max_output_tokens"))).classify(request())
    assert exc.value.error_type == "incomplete_response" and not exc.value.transient


def test_access_check_reports_the_model_or_a_fatal_error():
    assert classifier(FakeClient()).check_access() == "gpt-5.6-luna"
    with pytest.raises(ProviderError) as exc:
        classifier(FakeClient(error=api_error(openai.NotFoundError, 404))).check_access()
    assert exc.value.fatal


# --- output validation -------------------------------------------------------------------------


def check(output):
    text = output if isinstance(output, str) else json.dumps(output)
    return validate_output(text, request(), SCHEMA)


def test_valid_output_keeps_recommendations_and_sentiments_apart():
    output, issues = check(good_output())
    assert issues == []
    assert {b["brand_id"]: (b["sentiment"], b["recommendation"]) for b in output["brands"]} == {
        "hubspot": ("positive", "primary"), "pipedrive": ("negative", "conditional"),
        "close": ("neutral", "conditional")}
    assert [c["is_brand"] for c in output["candidates"]] == [True, False]


def test_evidence_may_differ_in_whitespace():
    out = good_output()
    out["brands"][0]["evidence"] = "HubSpot  is the\nbest pick"
    assert check(out)[1] == []


@pytest.mark.parametrize("mutate, issue", [
    (lambda o: o["brands"][0].update(sentiment="great"), "schema: 'great' is not one of"),
    (lambda o: o["brands"][0].pop("evidence"), "schema: 'evidence' is a required property"),
    (lambda o: o["brands"][0].update(extra=1), "schema: Additional properties"),
    (lambda o: o["brands"].pop(), "brand_id missing: ['close']"),
    (lambda o: o["brands"].append({**o["brands"][0], "brand_id": "attio"}), "brand_id not in the request: ['attio']"),
    (lambda o: o["candidates"].append(o["candidates"][0]), "candidate_id returned more than once"),
    (lambda o: o["brands"][0].update(evidence="HubSpot is loved by everyone"), "not a quote from the answer"),
    (lambda o: o["brands"][0].update(evidence=" "), "evidence is empty"),
    (lambda o: o["candidates"][1].update(evidence="any of them"), "does not contain 'Close'"),
])
def test_bad_output_is_rejected(mutate, issue):
    out = good_output()
    mutate(out)
    output, issues = check(out)
    assert output is None
    assert any(issue in i for i in issues), issues


def test_malformed_json_is_rejected():
    assert check('{"brands": [') == (None, ["output is not JSON: Expecting value"])


# --- cost ----------------------------------------------------------------------------------------


def test_cost_counts_cached_tokens_at_the_cached_price():
    price = Price(input=0.20, cached_input=0.02, output=1.20)
    # 200 uncached * 0.20 + 1000 cached * 0.02 + 300 output * 1.20, per million
    assert estimate_cost(price, 1200, 1000, 300) == pytest.approx((40 + 20 + 360) / 1e6)
    assert estimate_cost(None, 1200, 1000, 300) is None
    assert estimate_cost(price, None, None, None) is None


def test_configured_price_is_used_for_the_configured_model():
    assert CONFIG.pricing[CONFIG.llm.model] == Price(0.20, 0.02, 1.20)


# Real cases from the pilot: the model drops ** and may straighten quotes, but must not paraphrase.
MARKDOWN_ANSWER = ("For most growing B2B companies, I’d pick **HubSpot first**. Add **Pipedrive** if your needs "
                   "are mostly sales-pipeline management. Team already using monday.com heavily:** **monday CRM**")


@pytest.mark.parametrize("quote, ok", [
    ("Pipedrive if your needs are mostly sales-pipeline management", True),
    ("I'd pick HubSpot first", True),
    ("Team already using monday.com heavily: **monday CRM**", True),
    ("HubSpot is the better default", False),
])
def test_quotes_may_drop_markdown_but_not_change_words(quote, ok):
    req = ClassificationRequest(1, "q", MARKDOWN_ANSWER, [{"brand_id": "hubspot", "name": "HubSpot", "aliases": []}], [])
    out = {"brands": [{"brand_id": "hubspot", "sentiment": "positive", "recommendation": "primary", "evidence": quote}],
           "candidates": []}
    assert (validate_output(json.dumps(out), req, SCHEMA)[0] is not None) is ok
