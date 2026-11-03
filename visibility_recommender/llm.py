"""One LLM call per opportunity: the evidence pack in, recommendation JSON out."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

import httpx
from google import genai
from google.genai import errors, types

from visibility_collector.providers.base import ProviderError
from visibility_collector.providers.gemini_provider import COMPLETE_FINISH_REASONS, classify_api_error, response_text
from visibility_extractor.llm import LLMCall
from visibility_recommender.config import GeneratorConfig, RecommendationConfigError
from visibility_recommender.evidence import EvidencePack
from visibility_recommender.output import GenerationOutput

GAP_DESCRIPTIONS = {
    "competitor_recommended_more": "competitors are recommended more often than the target brand",
    "competitor_cited_more": "competitors' own websites are cited more often than the target brand's",
    "target_mentioned_less": "the target brand is mentioned less often than competitors",
    "weaker_on_provider": "the target brand is less visible on this provider than on the other one",
}

PROMPTS = {
    "r1": """\
You turn one measured visibility gap of a brand in AI-generated answers into concrete actions for that brand.

Rules:
- Use only the facts in "evidence". In evidence_ids, cite the ids of the facts each recommendation relies on, \
including at least one query, domain or citation id, not only metrics.
- Do not mention any number, percentage, domain, URL or source that is not in the facts. Do not invent statistics.
- Be specific: name the queries, topics, domains or cited pages the action targets. No generic marketing advice \
such as "create quality content" or "improve SEO".
- category is one of: owned_content_optimization (pages on the target brand's own site), editorial_outreach \
(coverage by third-party publishers, review or analyst sites), community_visibility (forums, video, social and \
Q&A communities), directory_listing (software directories and review listings).
- Write the rationale as a hypothesis: the facts show what appears together in answers, not what causes it. \
Never claim or imply a guaranteed outcome, and do not write that an action "will" increase visibility.
- expected_impact, effort and confidence are your estimates. Lower confidence when caveats apply or evidence is thin.
- Give 1 to 3 recommendations, strongest first, each a different action.
- If the facts support no specific action, set insufficient_evidence to true, explain why in \
insufficient_reason, and return no recommendations.
- Copy opportunity_id exactly.""",
}


def build_payload(pack: EvidencePack) -> str:
    o = pack.opportunity
    return json.dumps({
        "opportunity_id": pack.opportunity_id,
        "target_brand": o.target_brand,
        "gap": {"type": o.gap_type, "description": GAP_DESCRIPTIONS[o.gap_type], "metric": o.metric,
                "provider": o.provider, "intent": o.intent, "observed": o.observed, "caveats": list(o.caveats)},
        "evidence": pack.facts(),
    }, ensure_ascii=False, indent=1)


class Generator(Protocol):
    provider: str
    model: str

    def generate(self, pack: EvidencePack) -> LLMCall: ...


class GeminiGenerator:
    provider = "gemini"

    def __init__(self, client: Any, config: GeneratorConfig):
        if config.prompt_version not in PROMPTS:
            raise RecommendationConfigError([f"llm.prompt_version {config.prompt_version!r} is not one of {list(PROMPTS)}"])
        self._client = client
        self.config = config
        self.model = config.model
        self.instructions = PROMPTS[config.prompt_version]

    @classmethod
    def from_config(cls, config: GeneratorConfig) -> GeminiGenerator:
        from visibility_db.config import load_dotenv_file

        load_dotenv_file()
        key = os.environ.get(config.api_key_env)
        if not key:
            raise RecommendationConfigError([f"{config.api_key_env} is not set"])
        # attempts=1 disables SDK retries: the runner retries, so every attempt is counted.
        http_options = types.HttpOptions(timeout=config.timeout_seconds * 1000,
                                         retry_options=types.HttpRetryOptions(attempts=1))
        return cls(genai.Client(api_key=key, http_options=http_options), config)

    def request_config(self) -> Any:
        return types.GenerateContentConfig(
            system_instruction=self.instructions,
            response_mime_type="application/json",
            response_json_schema=GenerationOutput.model_json_schema(),
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )

    def check_access(self) -> str:
        """The model name the API reports. Raises ProviderError if it is not available."""
        try:
            return self._client.models.get(model=self.model).name
        except errors.APIError as e:
            raise classify_api_error(e, {"model": self.model}, {}) from None

    def generate(self, pack: EvidencePack) -> LLMCall:
        params = {"model": self.model, "prompt_version": self.config.prompt_version,
                  "temperature": self.config.temperature, "max_output_tokens": self.config.max_output_tokens}
        started = time.monotonic()
        try:
            response = self._client.models.generate_content(model=self.model, contents=build_payload(pack),
                                                            config=self.request_config())
        except errors.APIError as e:
            raise classify_api_error(e, params, {}) from None
        except httpx.TimeoutException as e:
            raise ProviderError("timeout", str(e) or "request timed out", transient=True, request_params=params) from None
        except httpx.TransportError as e:
            raise ProviderError("connection_error", str(e) or type(e).__name__, transient=True,
                                request_params=params) from None
        latency_ms = int((time.monotonic() - started) * 1000)

        candidate = response.candidates[0] if response.candidates else None
        finish_reason = candidate.finish_reason.name if candidate and candidate.finish_reason else None
        text = response_text(response)
        if finish_reason not in COMPLETE_FINISH_REASONS or not text.strip():
            block = response.prompt_feedback.block_reason if response.prompt_feedback else None
            raise ProviderError("incomplete_response",
                                f"finish_reason={finish_reason}, prompt block_reason={block}, text length {len(text)}",
                                transient=False, model_id=response.model_version, request_params=params)
        usage = response.usage_metadata
        thoughts = usage.thoughts_token_count if usage else None
        output = None
        if usage and (usage.candidates_token_count is not None or thoughts is not None):
            output = (usage.candidates_token_count or 0) + (thoughts or 0)  # thinking is billed as output
        return LLMCall(
            output_text=text,
            model_id=response.model_version,
            input_tokens=usage.prompt_token_count if usage else None,
            cached_input_tokens=usage.cached_content_token_count if usage else None,
            output_tokens=output,
            reasoning_tokens=thoughts,
            latency_ms=latency_ms,
        )


def make_generator(config: GeneratorConfig) -> GeminiGenerator:
    if config.provider == "gemini":
        return GeminiGenerator.from_config(config)
    raise RecommendationConfigError([f"unsupported provider {config.provider!r}"])
