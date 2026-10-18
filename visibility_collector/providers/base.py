"""Common interface for provider clients.

A client turns one collection request into either a ProviderResult or a
ProviderError. It makes exactly one API call per invocation: retries,
timing, and persistence belong to the runner, so every attempt is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

MAX_ERROR_MESSAGE_CHARS = 2000


@dataclass(frozen=True)
class CollectionTask:
    """What a provider needs to answer one collection request."""

    request_id: str
    query_id: str
    query_text: str
    provider: str
    run_number: int
    market: str
    language: str
    search_enabled: bool


@dataclass(frozen=True)
class ProviderResult:
    model_id: str | None
    response_text: str
    request_params: dict[str, Any]
    search_config: dict[str, Any]
    raw_response: dict[str, Any]
    # Provider-native citation/grounding data, exactly as returned; None if the
    # API returned none. OpenAI and Gemini shapes are deliberately not unified.
    grounding_metadata: dict[str, Any] | None
    input_tokens: int | None
    output_tokens: int | None


class ProviderError(Exception):
    """One failed API call, classified for the retry loop.

    transient: worth retrying (rate limit, timeout, 5xx).
    fatal: retrying any request is pointless (bad key, no quota, unknown
           model), so the runner stops this provider instead of failing
           every remaining request.
    """

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        transient: bool,
        fatal: bool = False,
        retry_after: float | None = None,
        model_id: str | None = None,
        request_params: dict[str, Any] | None = None,
        search_config: dict[str, Any] | None = None,
        raw_response: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.message = (message or error_type)[:MAX_ERROR_MESSAGE_CHARS]
        self.transient = transient
        self.fatal = fatal
        self.retry_after = retry_after
        self.model_id = model_id
        self.request_params = request_params or {}
        self.search_config = search_config or {}
        self.raw_response = raw_response


class ProviderClient(Protocol):
    name: str
    model: str

    def generate(self, task: CollectionTask) -> ProviderResult: ...


def parse_retry_after(headers: Any) -> float | None:
    """Seconds from a Retry-After header, if present and numeric."""
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None
