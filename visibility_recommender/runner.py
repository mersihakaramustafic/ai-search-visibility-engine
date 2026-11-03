"""One opportunity from evidence pack to validated recommendations, with retries."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from visibility_collector.providers.base import ProviderError
from visibility_extractor.llm import estimate_cost
from visibility_extractor.runner import _add, _as_json, backoff
from visibility_recommender.config import RecommendationConfig
from visibility_recommender.evidence import EvidencePack
from visibility_recommender.llm import Generator
from visibility_recommender.output import GenerationOutput, validate_output


@dataclass(frozen=True)
class Generation:
    opportunity_id: str
    generator_version: str
    status: str  # 'succeeded' | 'insufficient_evidence' | 'failed'
    provider: str
    model_requested: str
    prompt_version: str
    schema_version: str
    output: GenerationOutput | None = None
    raw_output: Any = None  # the applied output, or the last rejected one
    insufficient_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    model_id: str | None = None
    attempts: int = 0
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_ms: int | None = None
    estimated_cost_usd: float | None = None


class RecommendationRunner:
    def __init__(self, generator: Generator, config: RecommendationConfig, brand_domains: set[str],
                 sleep: Callable[[float], None] = time.sleep, jitter: Callable[[], float] = random.random):
        self.generator = generator
        self.config = config
        self.brand_domains = brand_domains
        self._sleep = sleep
        self._jitter = jitter

    def process(self, pack: EvidencePack) -> Generation:
        """Raises ProviderError only when the error is fatal for every opportunity (bad key, unknown model)."""
        llm = self.config.llm
        g = Generation(pack.opportunity_id, self.config.generator_version, "failed", self.generator.provider,
                       self.generator.model, llm.prompt_version, llm.schema_version)
        if pack.insufficient_reason:
            return replace(g, status="insufficient_evidence", insufficient_reason=pack.insufficient_reason)

        retry = self.config.retry
        for attempt in range(1, retry.max_attempts + 1):
            try:
                call = self.generator.generate(pack)
            except ProviderError as e:
                g = replace(g, attempts=attempt, model_id=e.model_id or g.model_id,
                          error_type=e.error_type, error_message=e.message)
                if e.fatal:
                    raise
                if not e.transient or attempt == retry.max_attempts:
                    return g
                self._sleep(backoff(retry, attempt, e.retry_after, self._jitter()))
                continue

            g = replace(g, attempts=attempt, model_id=call.model_id, raw_output=_as_json(call.output_text),
                      input_tokens=_add(g.input_tokens, call.input_tokens),
                      cached_input_tokens=_add(g.cached_input_tokens, call.cached_input_tokens),
                      output_tokens=_add(g.output_tokens, call.output_tokens),
                      reasoning_tokens=_add(g.reasoning_tokens, call.reasoning_tokens),
                      latency_ms=_add(g.latency_ms, call.latency_ms))
            g = replace(g, estimated_cost_usd=estimate_cost(self.config.price, g.input_tokens, g.cached_input_tokens,
                                                          g.output_tokens))
            output, errors = validate_output(call.output_text, pack, self.brand_domains)
            if output is None:
                g = replace(g, error_type="invalid_output", error_message="; ".join(errors)[:2000],
                          validation_errors=errors)
                continue  # an invalid answer is retried at once: nothing to wait for
            status = "insufficient_evidence" if output.insufficient_evidence else "succeeded"
            return replace(g, status=status, output=output, insufficient_reason=output.insufficient_reason,
                         error_type=None, error_message=None, validation_errors=[])
        return g
