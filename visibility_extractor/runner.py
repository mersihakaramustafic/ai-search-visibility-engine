"""One observation from start to finish: rules, then (if the strategy says so) the LLM, with retries."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from visibility_collector.config import RetryConfig
from visibility_collector.providers.base import ProviderError
from visibility_extractor.config import ExtractionConfig
from visibility_extractor.llm import LLMClassifier, build_request, estimate_cost, validate_output
from visibility_extractor.matching import aggregate_mentions
from visibility_extractor.pipeline import Extraction, LLMRun, Observation, RulesExtractor, validate


def backoff(retry: RetryConfig, attempt: int, retry_after: float | None, jitter: float) -> float:
    """Exponential backoff with jitter, never shorter than the server's Retry-After."""
    base = min(retry.max_backoff_seconds, retry.initial_backoff_seconds * 2 ** (attempt - 1))
    delay = base * (0.5 + jitter / 2)
    if retry_after is not None:
        delay = max(delay, min(retry_after, retry.max_backoff_seconds))
    return delay


def _as_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _add(total: int | None, value: int | None) -> int | None:
    return value if total is None else total + (value or 0)


class ExtractionRunner:
    def __init__(self, rules: RulesExtractor, config: ExtractionConfig, classifier: LLMClassifier | None = None,
                 sleep: Callable[[float], None] = time.sleep, jitter: Callable[[], float] = random.random):
        self.rules = rules
        self.config = config
        self.classifier = classifier
        self._sleep = sleep
        self._jitter = jitter

    def needs_llm(self, obs: Observation, e: Extraction) -> bool:
        return (self.config.strategy_for(obs.provider) == "rules_and_llm" and e.status == "succeeded"
                and e.validation_status != "invalid" and self._request(obs, e) is not None)

    def _request(self, obs: Observation, e: Extraction):
        return build_request(obs.observation_id, obs.query_text, obs.response_text, self.rules.brands,
                             e.mentions, e.candidates, self.config.matching.snippet_chars)

    def process(self, obs: Observation) -> Extraction:
        e = replace(self.rules.extract(obs), strategy=self.config.strategy_for(obs.provider))
        if e.strategy == "rules_only" or e.status == "failed" or e.validation_status == "invalid":
            return e
        llm = self.config.llm
        base = LLMRun("not_needed", llm.model, llm.prompt_version, llm.schema_version)
        request = self._request(obs, e)
        if request is None:
            return replace(e, llm=base, target_brand_recommended=False)
        return self._classify(obs, e, request, base)

    def _classify(self, obs: Observation, e: Extraction, request, run: LLMRun) -> Extraction:
        price = self.config.pricing.get(run.model_requested)
        retry = self.config.retry
        error_type, error_message, issues = "", "", []
        for attempt in range(1, retry.max_attempts + 1):
            run = replace(run, attempts=attempt)
            try:
                call = self.classifier.classify(request)
            except ProviderError as err:
                if err.fatal:
                    raise
                error_type, error_message = err.error_type, err.message
                run = replace(run, model_id=err.model_id or run.model_id)
                if not err.transient or attempt == retry.max_attempts:
                    break
                self._sleep(backoff(retry, attempt, err.retry_after, self._jitter()))
                continue

            cost = estimate_cost(price, call.input_tokens, call.cached_input_tokens, call.output_tokens)
            run = replace(
                run, model_id=call.model_id,
                input_tokens=_add(run.input_tokens, call.input_tokens),
                cached_input_tokens=_add(run.cached_input_tokens, call.cached_input_tokens),
                output_tokens=_add(run.output_tokens, call.output_tokens),
                reasoning_tokens=_add(run.reasoning_tokens, call.reasoning_tokens),
                estimated_cost_usd=None if cost is None else round((run.estimated_cost_usd or 0) + cost, 8),
                latency_ms=_add(run.latency_ms, call.latency_ms),
            )
            output, issues = validate_output(call.output_text, request, self.classifier.schema)
            if output is not None:
                return self._apply(obs, e, output, replace(run, status="succeeded", output=output))
            run = replace(run, output=_as_json(call.output_text))
            error_type, error_message = "invalid_output", "; ".join(issues)[:500]

        return Extraction(obs.observation_id, e.extractor_version, "failed", None, validation_issues=issues,
                          error_type=error_type, error_message=error_message or error_type,
                          strategy=e.strategy, llm=replace(run, status="failed"))

    def _apply(self, obs: Observation, e: Extraction, output: dict[str, Any], run: LLMRun) -> Extraction:
        accepted = {c["candidate_id"] for c in output["candidates"] if c["is_brand"]}
        occurrences = e.confirmed + [o for n, o in enumerate(e.candidates, start=1) if n in accepted]
        labels: dict[str, dict[str, Any]] = {b["brand_id"]: b for b in output["brands"]}
        mentions = [replace(m, sentiment=labels[m.brand_id]["sentiment"],
                            recommendation=labels[m.brand_id]["recommendation"],
                            classification_evidence=labels[m.brand_id]["evidence"])
                    for m in aggregate_mentions(obs.response_text, occurrences, self.config.matching.snippet_chars)]

        errors = validate(obs.response_text, mentions, e.citations, self.rules.brand_ids)
        if errors:
            return replace(e, validation_status="invalid", validation_issues=errors, mentions=[], citations=[],
                           llm=run)
        target = self.rules.target_ids
        mentioned = {m.brand_id for m in mentions}
        return replace(
            e, mentions=mentions, llm=run,
            target_brand_mentioned=bool(mentioned & target),
            competitor_brands_mentioned=len(mentioned - target),
            target_brand_recommended=any(m.brand_id in target and m.recommendation != "none" for m in mentions),
        )
