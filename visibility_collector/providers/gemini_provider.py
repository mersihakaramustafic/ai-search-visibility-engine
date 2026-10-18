"""Gemini client using google-genai, with Grounding with Google Search when enabled."""

from __future__ import annotations

from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from visibility_collector.config import ProviderConfig
from visibility_collector.providers.base import CollectionTask, ProviderError, ProviderResult, parse_retry_after

TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
FATAL_STATUS = {401, 403, 404}
# Finish reasons that still mean a complete answer.
COMPLETE_FINISH_REASONS = {"STOP"}


class GeminiProvider:
    name = "gemini"

    def __init__(self, client: Any, model: str, max_output_tokens: int):
        self._client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    @classmethod
    def from_config(cls, config: ProviderConfig) -> GeminiProvider:
        # attempts=1 disables SDK retries: the runner retries, so every attempt is recorded.
        http_options = types.HttpOptions(
            timeout=int(config.timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        client = genai.Client(api_key=config.api_key(), http_options=http_options)
        return cls(client, config.model, config.max_output_tokens)

    def build_request(self, task: CollectionTask) -> tuple[dict[str, Any], dict[str, Any], types.GenerateContentConfig]:
        params: dict[str, Any] = {
            "model": self.model,
            "contents": task.query_text,
            "max_output_tokens": self.max_output_tokens,
        }
        search_config: dict[str, Any] = {"enabled": task.search_enabled}
        tools = None
        if task.search_enabled:
            tools = [types.Tool(google_search=types.GoogleSearch())]
            search_config["tool"] = "google_search"
            # The API has no market/location parameter for search grounding.
            search_config["location"] = None
        config = types.GenerateContentConfig(max_output_tokens=self.max_output_tokens, tools=tools)
        return params, search_config, config

    def generate(self, task: CollectionTask) -> ProviderResult:
        params, search_config, config = self.build_request(task)
        try:
            response = self._client.models.generate_content(model=self.model, contents=task.query_text, config=config)
        except errors.APIError as e:
            raise classify_api_error(e, params, search_config) from None
        except httpx.TimeoutException as e:
            raise ProviderError("timeout", str(e) or "request timed out", transient=True,
                                request_params=params, search_config=search_config) from None
        except httpx.TransportError as e:
            raise ProviderError("connection_error", str(e) or type(e).__name__, transient=True,
                                request_params=params, search_config=search_config) from None

        # sdk_http_response holds HTTP headers, not model output.
        raw = response.model_dump(mode="json", exclude_none=True, exclude={"sdk_http_response"})
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = candidate.finish_reason.name if candidate and candidate.finish_reason else None
        text = response_text(response)
        if finish_reason not in COMPLETE_FINISH_REASONS or not text.strip():
            block = response.prompt_feedback.block_reason if response.prompt_feedback else None
            raise ProviderError(
                "incomplete_response",
                f"finish_reason={finish_reason}, prompt block_reason={block}, text length {len(text)}",
                transient=False,
                model_id=response.model_version,
                request_params=params,
                search_config=search_config,
                raw_response=raw,
            )

        usage = response.usage_metadata
        output_tokens = None
        if usage and (usage.candidates_token_count is not None or usage.thoughts_token_count is not None):
            # Thinking tokens are billed as output, as OpenAI's output_tokens includes reasoning.
            output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return ProviderResult(
            model_id=response.model_version,
            response_text=text,
            request_params=params,
            search_config=search_config,
            raw_response=raw,
            grounding_metadata=extract_grounding(raw),
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=output_tokens,
        )


def response_text(response: types.GenerateContentResponse) -> str:
    """Join the text parts of the first candidate, skipping thought parts."""
    candidate = response.candidates[0] if response.candidates else None
    if not candidate or not candidate.content or not candidate.content.parts:
        return ""
    return "".join(p.text for p in candidate.content.parts if p.text and not p.thought)


def extract_grounding(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Keep the first candidate's grounding and citation metadata as returned."""
    candidates = raw.get("candidates") or []
    if not candidates:
        return None
    metadata = {k: candidates[0][k] for k in ("grounding_metadata", "citation_metadata") if candidates[0].get(k)}
    return metadata or None


def classify_api_error(e: errors.APIError, params: dict[str, Any], search_config: dict[str, Any]) -> ProviderError:
    code = e.code
    status = getattr(e, "status", None)
    # An invalid key comes back as 400 INVALID_ARGUMENT with reason API_KEY_INVALID.
    fatal = code in FATAL_STATUS or "API_KEY_INVALID" in str(e.details)
    transient = not fatal and code in TRANSIENT_STATUS
    retry_after = parse_retry_after(e.response.headers) if getattr(e, "response", None) is not None else None
    return ProviderError(
        f"{type(e).__name__}:{code}" + (f":{status}" if status else ""),
        str(e),
        transient=transient,
        fatal=fatal,
        retry_after=retry_after,
        request_params=params,
        search_config=search_config,
        raw_response={"status_code": code, "body": e.details} if getattr(e, "details", None) else None,
    )
