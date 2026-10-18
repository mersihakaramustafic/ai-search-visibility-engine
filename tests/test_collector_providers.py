"""Provider clients against mocked SDK responses and errors (no network)."""

from __future__ import annotations

import copy

import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from google.genai import types
from openai.types.responses import Response

from visibility_collector.providers.base import CollectionTask, ProviderError
from visibility_collector.providers.gemini_provider import GeminiProvider
from visibility_collector.providers.openai_provider import OpenAIProvider

TASK = CollectionTask(
    request_id="REQ-Q001-openai-R1", query_id="Q001", query_text="What is the best CRM?",
    provider="openai", run_number=1, market="US", language="en", search_enabled=True,
)
NO_SEARCH = CollectionTask(**{**vars(TASK), "search_enabled": False})

OPENAI_RESPONSE = {
    "id": "resp_1", "object": "response", "created_at": 1760000000, "model": "gpt-5.5-2026-04-23",
    "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [{"type": "web_search"}],
    "output": [
        {"type": "web_search_call", "id": "ws_1", "status": "completed",
         "action": {"type": "search", "query": "best crm", "sources": [{"type": "url", "url": "https://example.com/a"}]}},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Try HubSpot.",
                      "annotations": [{"type": "url_citation", "url": "https://example.com/a", "title": "A",
                                       "start_index": 0, "end_index": 12}]}]},
    ],
    "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
              "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 30},
}

GEMINI_RESPONSE = {
    "candidates": [{
        "content": {"role": "model", "parts": [{"text": "hidden reasoning", "thought": True},
                                               {"text": "HubSpot is popular."}]},
        "finish_reason": "STOP",
        "grounding_metadata": {
            "web_search_queries": ["best crm"],
            "grounding_chunks": [{"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/x",
                                          "title": "example.com", "domain": "example.com"}}],
            "grounding_supports": [{"segment": {"start_index": 0, "end_index": 19, "text": "HubSpot is popular."},
                                    "grounding_chunk_indices": [0]}],
        },
    }],
    "usage_metadata": {"prompt_token_count": 8, "candidates_token_count": 12, "thoughts_token_count": 30,
                       "total_token_count": 50},
    "model_version": "gemini-3.5-flash",
}


class FakeOpenAI:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return Response.model_validate(self.result)


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


def openai_status_error(cls, status, body, headers=None):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls("error message", response=response, body=body)


# --- OpenAI ------------------------------------------------------------------


def test_openai_success_with_search_records_citations_and_usage():
    fake = FakeOpenAI(OPENAI_RESPONSE)
    result = OpenAIProvider(fake, "gpt-5.5-2026-04-23", 16000).generate(TASK)

    sent = fake.calls[0]
    assert sent["input"] == TASK.query_text
    assert sent["tools"] == [{"type": "web_search", "user_location": {"type": "approximate", "country": "US"}}]
    assert sent["include"] == ["web_search_call.action.sources"]
    assert result.model_id == "gpt-5.5-2026-04-23"
    assert result.response_text == "Try HubSpot."
    assert (result.input_tokens, result.output_tokens) == (10, 20)
    assert result.search_config["enabled"] is True
    assert result.grounding_metadata["url_citations"][0]["url"] == "https://example.com/a"
    assert result.grounding_metadata["web_search_calls"][0]["action"]["query"] == "best crm"
    assert result.raw_response["id"] == "resp_1"


def test_openai_without_search_sends_no_tools_and_has_no_grounding():
    body = copy.deepcopy(OPENAI_RESPONSE)
    body["output"] = [body["output"][1]]
    body["output"][0]["content"][0]["annotations"] = []
    fake = FakeOpenAI(body)
    result = OpenAIProvider(fake, "gpt-5.5", 16000).generate(NO_SEARCH)
    assert "tools" not in fake.calls[0] and "include" not in fake.calls[0]
    assert result.search_config == {"enabled": False}
    assert result.grounding_metadata is None


def test_openai_market_location_can_be_disabled():
    fake = FakeOpenAI(OPENAI_RESPONSE)
    OpenAIProvider(fake, "gpt-5.5", 16000, use_market_location=False).generate(TASK)
    assert fake.calls[0]["tools"] == [{"type": "web_search"}]


def test_openai_incomplete_response_is_a_permanent_error_with_raw_kept():
    body = copy.deepcopy(OPENAI_RESPONSE)
    body["status"] = "incomplete"
    body["incomplete_details"] = {"reason": "max_output_tokens"}
    with pytest.raises(ProviderError) as exc:
        OpenAIProvider(FakeOpenAI(body), "gpt-5.5", 100).generate(TASK)
    assert exc.value.error_type == "incomplete_response"
    assert not exc.value.transient and not exc.value.fatal
    assert "max_output_tokens" in exc.value.message
    assert exc.value.raw_response["status"] == "incomplete"


@pytest.mark.parametrize("error, transient, fatal", [
    (openai_status_error(openai.RateLimitError, 429, {"code": "rate_limit_exceeded"}, {"retry-after": "7"}), True, False),
    (openai_status_error(openai.RateLimitError, 429, {"code": "insufficient_quota"}), False, True),
    # Seen live: no credits left. Must stop the provider, not be retried as a rate limit.
    (openai_status_error(openai.RateLimitError, 429, {"code": "credit_balance_exhausted",
                                                      "type": "insufficient_quota"}), False, True),
    (openai_status_error(openai.RateLimitError, 429, {"code": "some_new_code", "type": "insufficient_quota"}),
     False, True),
    (openai_status_error(openai.InternalServerError, 503, None), True, False),
    (openai_status_error(openai.AuthenticationError, 401, {"code": "invalid_api_key"}), False, True),
    (openai_status_error(openai.NotFoundError, 404, {"code": "model_not_found"}), False, True),
    (openai_status_error(openai.BadRequestError, 400, {"code": "invalid_value"}), False, False),
    (openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")), True, False),
    (openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")), True, False),
])
def test_openai_error_classification(error, transient, fatal):
    with pytest.raises(ProviderError) as exc:
        OpenAIProvider(FakeOpenAI(error), "gpt-5.5", 16000).generate(TASK)
    assert (exc.value.transient, exc.value.fatal) == (transient, fatal)
    assert exc.value.request_params["model"] == "gpt-5.5"


def test_openai_rate_limit_keeps_retry_after():
    error = openai_status_error(openai.RateLimitError, 429, {"code": "rate_limit_exceeded"}, {"retry-after": "7"})
    with pytest.raises(ProviderError) as exc:
        OpenAIProvider(FakeOpenAI(error), "gpt-5.5", 16000).generate(TASK)
    assert exc.value.retry_after == 7.0
    assert exc.value.error_type == "RateLimitError:rate_limit_exceeded"


# --- Gemini ------------------------------------------------------------------


def test_gemini_success_with_grounding():
    fake = FakeGemini(GEMINI_RESPONSE)
    result = GeminiProvider(fake, "gemini-3.5-flash", 16000).generate(TASK)

    config = fake.calls[0]["config"]
    assert config.tools[0].google_search is not None
    assert fake.calls[0]["contents"] == TASK.query_text
    assert result.model_id == "gemini-3.5-flash"
    assert result.response_text == "HubSpot is popular."  # thought part excluded
    assert result.input_tokens == 8
    assert result.output_tokens == 42  # candidates + thoughts, both billed as output
    chunk = result.grounding_metadata["grounding_metadata"]["grounding_chunks"][0]["web"]
    assert chunk["domain"] == "example.com"  # stored as returned, redirect URI untouched
    assert chunk["uri"].startswith("https://vertexaisearch.cloud.google.com/")
    assert result.search_config == {"enabled": True, "tool": "google_search", "location": None}


def test_gemini_without_search_sends_no_tools():
    body = copy.deepcopy(GEMINI_RESPONSE)
    del body["candidates"][0]["grounding_metadata"]
    fake = FakeGemini(body)
    result = GeminiProvider(fake, "gemini-3.5-flash", 16000).generate(NO_SEARCH)
    assert fake.calls[0]["config"].tools is None
    assert result.grounding_metadata is None


@pytest.mark.parametrize("finish_reason", ["SAFETY", "MAX_TOKENS"])
def test_gemini_incomplete_answer_is_permanent_error(finish_reason):
    body = copy.deepcopy(GEMINI_RESPONSE)
    body["candidates"][0]["finish_reason"] = finish_reason
    with pytest.raises(ProviderError) as exc:
        GeminiProvider(FakeGemini(body), "gemini-3.5-flash", 16000).generate(TASK)
    assert exc.value.error_type == "incomplete_response"
    assert finish_reason in exc.value.message
    assert exc.value.raw_response["model_version"] == "gemini-3.5-flash"


def test_gemini_empty_candidates_is_permanent_error():
    body = {"candidates": [], "prompt_feedback": {"block_reason": "SAFETY"}, "model_version": "gemini-3.5-flash"}
    with pytest.raises(ProviderError) as exc:
        GeminiProvider(FakeGemini(body), "gemini-3.5-flash", 16000).generate(TASK)
    assert "SAFETY" in exc.value.message


def gemini_error(cls, code, status, reason=None):
    error = {"code": code, "message": "error message", "status": status}
    if reason:
        error["details"] = [{"reason": reason}]
    return cls(code, {"error": error})


@pytest.mark.parametrize("error, transient, fatal", [
    (gemini_error(genai_errors.ClientError, 429, "RESOURCE_EXHAUSTED"), True, False),
    (gemini_error(genai_errors.ServerError, 503, "UNAVAILABLE"), True, False),
    (gemini_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT", "API_KEY_INVALID"), False, True),
    (gemini_error(genai_errors.ClientError, 403, "PERMISSION_DENIED"), False, True),
    (gemini_error(genai_errors.ClientError, 404, "NOT_FOUND"), False, True),
    (gemini_error(genai_errors.ClientError, 400, "INVALID_ARGUMENT"), False, False),
    (httpx.ReadTimeout("timed out"), True, False),
    (httpx.ConnectError("refused"), True, False),
])
def test_gemini_error_classification(error, transient, fatal):
    with pytest.raises(ProviderError) as exc:
        GeminiProvider(FakeGemini(error), "gemini-3.5-flash", 16000).generate(TASK)
    assert (exc.value.transient, exc.value.fatal) == (transient, fatal)
