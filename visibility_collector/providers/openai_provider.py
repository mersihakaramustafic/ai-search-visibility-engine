"""OpenAI client using the Responses API, with the web_search tool when enabled."""

from __future__ import annotations

from typing import Any

import openai

from visibility_collector.config import ProviderConfig
from visibility_collector.providers.base import CollectionTask, ProviderError, ProviderResult, parse_retry_after

# Return the full list of sources each search consulted, not only the cited ones.
SEARCH_INCLUDE = ["web_search_call.action.sources"]
FATAL_ERROR_CODES = {"insufficient_quota", "credit_balance_exhausted", "invalid_api_key", "model_not_found"}
# Billing errors share this type even when the code is new, e.g.
# {"type": "insufficient_quota", "code": "credit_balance_exhausted"}.
FATAL_ERROR_TYPES = {"insufficient_quota"}


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: Any, model: str, max_output_tokens: int, use_market_location: bool = True):
        self._client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.use_market_location = use_market_location

    @classmethod
    def from_config(cls, config: ProviderConfig) -> OpenAIProvider:
        # max_retries=0: the runner retries, so every attempt becomes an observation.
        client = openai.OpenAI(api_key=config.api_key(), timeout=config.timeout_seconds, max_retries=0)
        web_search = config.options.get("web_search", {})
        return cls(client, config.model, config.max_output_tokens, web_search.get("use_market_location", True))

    def build_request(self, task: CollectionTask) -> tuple[dict[str, Any], dict[str, Any]]:
        params: dict[str, Any] = {
            "model": self.model,
            "input": task.query_text,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        search_config: dict[str, Any] = {"enabled": task.search_enabled}
        if task.search_enabled:
            tool: dict[str, Any] = {"type": "web_search"}
            if self.use_market_location and task.market:
                tool["user_location"] = {"type": "approximate", "country": task.market}
            params["tools"] = [tool]
            params["include"] = SEARCH_INCLUDE
            search_config.update(tool=tool, include=SEARCH_INCLUDE)
        return params, search_config

    def generate(self, task: CollectionTask) -> ProviderResult:
        params, search_config = self.build_request(task)
        try:
            response = self._client.responses.create(**params)
        except openai.APIError as e:
            raise classify_error(e, params, search_config) from None

        raw = response.model_dump(mode="json")
        text = response.output_text or ""
        if response.status != "completed" or not text.strip():
            reason = (response.incomplete_details.reason if response.incomplete_details else None) or response.status
            raise ProviderError(
                "incomplete_response",
                f"Response status {response.status!r} (reason: {reason}); text length {len(text)}",
                transient=False,
                model_id=response.model,
                request_params=params,
                search_config=search_config,
                raw_response=raw,
            )
        usage = response.usage
        return ProviderResult(
            model_id=response.model,
            response_text=text,
            request_params=params,
            search_config=search_config,
            raw_response=raw,
            grounding_metadata=extract_grounding(raw),
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
        )


def extract_grounding(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Collect the url_citation annotations and web_search_call items as returned."""
    citations = []
    searches = []
    for item in raw.get("output") or []:
        if item.get("type") == "web_search_call":
            searches.append(item)
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                for annotation in part.get("annotations") or []:
                    if annotation.get("type") == "url_citation":
                        citations.append(annotation)
    if not citations and not searches:
        return None
    return {"url_citations": citations, "web_search_calls": searches}


def classify_error(e: openai.APIError, params: dict[str, Any], search_config: dict[str, Any]) -> ProviderError:
    status = getattr(e, "status_code", None)
    code = getattr(e, "code", None)
    retry_after = parse_retry_after(e.response.headers) if getattr(e, "response", None) is not None else None
    error_kind = e.body.get("type") if isinstance(e.body, dict) else None
    fatal = (code in FATAL_ERROR_CODES or error_kind in FATAL_ERROR_TYPES
             or isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError)))
    if isinstance(e, openai.NotFoundError):
        fatal = True  # unknown model or endpoint: every request would fail the same way
    transient = not fatal and (
        isinstance(e, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError))
        or status in (408, 409)
        or (status is not None and status >= 500)
    )
    error_type = type(e).__name__ + (f":{code}" if code else "")
    return ProviderError(
        error_type,
        str(e),
        transient=transient,
        fatal=fatal,
        retry_after=retry_after,
        request_params=params,
        search_config=search_config,
        raw_response={"status_code": status, "body": e.body} if status is not None else None,
    )
