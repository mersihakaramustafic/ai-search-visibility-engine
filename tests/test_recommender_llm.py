"""Recommendation config and the Gemini client, with a fake SDK client (no API calls)."""

from __future__ import annotations

import json

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from recommender_fixtures import pack
from test_collector_providers import gemini_error
from visibility_collector.providers.base import ProviderError
from visibility_recommender.config import RecommendationConfigError, load_config, parse_config
from visibility_recommender.llm import PROMPTS, GeminiGenerator, build_payload

CONFIG = load_config()
PACK = pack()


class FakeGemini:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return types.GenerateContentResponse.model_validate(self.result)

    def get(self, model):
        if isinstance(self.result, Exception):
            raise self.result
        return types.Model(name=f"models/{model}")


def response(text='{"ok": true}', finish_reason="STOP", usage=None):
    return {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": text}]},
                            "finish_reason": finish_reason}],
            "usage_metadata": usage or {"prompt_token_count": 1200, "cached_content_token_count": 200,
                                        "candidates_token_count": 300, "thoughts_token_count": 500},
            "model_version": "gemini-3.8-flash"}


def generator(result):
    fake = FakeGemini(result)
    return GeminiGenerator(fake, CONFIG.llm), fake


def test_config_defaults():
    assert (CONFIG.generator_version, CONFIG.llm.provider, CONFIG.llm.model) == ("recs-v2", "gemini", "gemini-3.8-flash")
    assert CONFIG.llm.prompt_version in PROMPTS and CONFIG.price is not None


def test_model_can_be_overridden_by_env():
    raw = {"generator_version": "recs-v2", "llm": {**vars(CONFIG.llm)}}
    assert parse_config(raw, {"RECOMMENDATION_MODEL": "gemini-3.5-flash"}).llm.model == "gemini-3.5-flash"


@pytest.mark.parametrize("change, message", [
    ({"provider": "openai"}, "llm.provider"),
    ({"temperature": 3}, "llm.temperature"),
    ({"max_output_tokens": 0}, "llm.max_output_tokens"),
    ({"model": "bad model"}, "llm.model"),
])
def test_config_errors(change, message):
    raw = {"generator_version": "recs-v2", "llm": {**vars(CONFIG.llm), **change}}
    with pytest.raises(RecommendationConfigError, match=message):
        parse_config(raw)


def test_payload_holds_the_gap_and_only_the_pack_facts():
    payload = json.loads(build_payload(PACK))
    assert payload["opportunity_id"] == PACK.opportunity_id
    assert payload["gap"]["type"] == "competitor_cited_more" and payload["gap"]["provider"] == "openai"
    assert payload["evidence"] == PACK.facts()


def test_request_uses_prompt_schema_and_settings():
    gen, fake = generator(response())
    call = gen.generate(PACK)
    sent = fake.calls[0]
    assert sent["model"] == "gemini-3.8-flash" and json.loads(sent["contents"])["opportunity_id"] == PACK.opportunity_id
    config = sent["config"]
    assert config.system_instruction == PROMPTS["r1"] and config.response_mime_type == "application/json"
    assert config.response_json_schema["title"] == "GenerationOutput"
    assert (config.temperature, config.max_output_tokens) == (0.2, 8000)
    assert call.output_text == '{"ok": true}'  # thought parts are skipped
    assert (call.model_id, call.input_tokens, call.cached_input_tokens, call.output_tokens, call.reasoning_tokens) == (
        "gemini-3.8-flash", 1200, 200, 800, 500)


@pytest.mark.parametrize("body", [response(finish_reason="MAX_TOKENS"), response(text="  "),
                                  {"candidates": [], "prompt_feedback": {"block_reason": "SAFETY"}}])
def test_incomplete_response_is_a_permanent_error(body):
    with pytest.raises(ProviderError) as exc:
        generator(body)[0].generate(PACK)
    assert (exc.value.error_type, exc.value.transient, exc.value.fatal) == ("incomplete_response", False, False)


@pytest.mark.parametrize("error, transient, fatal", [
    (gemini_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED"), True, False),
    (gemini_error(genai_errors.ServerError, 503, "UNAVAILABLE"), True, False),
    (gemini_error(genai_errors.ClientError, 404, "NOT_FOUND"), False, True),
    (gemini_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "API_KEY_INVALID"), False, True),
    (httpx.ReadTimeout("timed out"), True, False),
])
def test_api_errors_are_classified(error, transient, fatal):
    with pytest.raises(ProviderError) as exc:
        generator(error)[0].generate(PACK)
    assert (exc.value.transient, exc.value.fatal) == (transient, fatal)


def test_check_access():
    assert generator(response())[0].check_access() == "models/gemini-3.8-flash"
    with pytest.raises(ProviderError) as exc:
        generator(gemini_error(genai_errors.ClientError, 404, "NOT_FOUND"))[0].check_access()
    assert exc.value.fatal
