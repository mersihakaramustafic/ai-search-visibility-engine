"""LLM classification of one answer: sentiment, recommendation, and ambiguous aliases.
One API call per answer.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from jsonschema import Draft202012Validator

from visibility_collector.providers.base import ProviderError
from visibility_collector.providers.openai_provider import classify_error
from visibility_extractor.config import ExtractionConfigError, LLMConfig, Price
from visibility_extractor.matching import Brand, MentionSummary, Occurrence, snippet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "brand_classification.schema.json"
MAX_EVIDENCE_CHARS = 300

_CANDIDATE_RULES = """\
For each item in "candidates" (an ambiguous word at a character offset in the answer):
- is_brand: true only if that occurrence names the brand's product or company;
  false for ordinary words such as "close a deal" or "every Monday".
- evidence: an exact quote from the answer that contains that occurrence.

Copy quotes character for character. Return every brand and every candidate exactly once."""

# Selected by llm.prompt_version. Old versions stay so stored results remain reproducible.
PROMPTS = {
    "p1": """\
You classify how one AI-generated answer talks about software brands.
Use only the answer text you are given, never outside knowledge about the brands.

For each item in "brands":
- sentiment: the answer's tone towards that brand. Use neutral unless it is clearly positive or clearly negative.
- recommendation:
    primary      presented as a top pick for the user's question
    conditional  recommended only for a stated situation or type of user
    none         only mentioned, listed without endorsement, used for comparison, or advised against
- evidence: the shortest exact quote from the answer that supports both labels.

""" + _CANDIDATE_RULES,
    "p2": """\
You classify how one AI-generated answer talks about software brands.
Use only the answer text you are given, never outside knowledge about the brands.

For each item in "brands":
- sentiment: the answer's tone towards that brand. Use neutral unless it is clearly positive or clearly negative.
- recommendation, judged against the user's question as asked:
    primary      the answer names it as its overall, default or first choice for that question,
                 without limiting it to a particular situation, segment or type of user
    conditional  recommended only for a stated situation, segment or type of user, e.g. "choose X if ...",
                 "best for enterprises", or one entry in a list of "best for <segment>" picks
    none         only mentioned, described, listed without endorsement, used for comparison, or advised against
  If unsure between primary and conditional, choose conditional.
- evidence: the shortest exact quote from the answer that supports both labels;
  prefer a quote that contains the brand's name.

""" + _CANDIDATE_RULES,
    "p3": """\
You classify how one AI-generated answer talks about software brands.
Use only the answer text you are given, never outside knowledge about the brands.

For each item in "brands":
- sentiment: the answer's tone towards that brand. Use neutral unless it is clearly positive or clearly negative.
- recommendation, judged against the user's question as asked:
    primary      the answer names it as its overall, default or first choice for that question,
                 without limiting it to a particular situation, segment or type of user
    conditional  recommended only for a stated situation, segment or type of user, e.g. "choose X if ...",
                 "best for enterprises", or one entry in a list of "best for <segment>" picks
    none         only mentioned, described, listed without endorsement, used for comparison, or advised against;
                 also when it is named only as an example of a category ("tools such as X, Y and Z"),
                 even if that category is recommended
  If unsure between primary and conditional, choose conditional.
- evidence: the shortest exact quote from the answer that supports both labels;
  prefer a quote that contains the brand's name.

""" + _CANDIDATE_RULES,
}


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ClassificationRequest:
    observation_id: int
    query_text: str
    response_text: str
    brands: list[dict[str, Any]]
    candidates: list[dict[str, Any]]

    def payload(self) -> str:
        return json.dumps({"question": self.query_text, "answer": self.response_text,
                           "brands": self.brands, "candidates": self.candidates}, ensure_ascii=False)


def build_request(observation_id: int, query_text: str, response_text: str, brands: list[Brand],
                  mentions: list[MentionSummary], candidates: list[Occurrence],
                  snippet_chars: int) -> ClassificationRequest | None:
    """None when there is nothing to classify, so no call is made."""
    order = [m.brand_id for m in mentions]
    order += [c.brand_id for c in candidates if c.brand_id not in order]
    if not order:
        return None
    by_id = {b.brand_id: b for b in brands}
    return ClassificationRequest(
        observation_id=observation_id,
        query_text=query_text,
        response_text=response_text,
        brands=[{"brand_id": i, "name": by_id[i].brand_name, "aliases": list(by_id[i].aliases)}
                for i in dict.fromkeys(order)],
        candidates=[{"candidate_id": n, "brand_id": c.brand_id, "text": c.text, "char_offset": c.start,
                     "context": snippet(response_text, c.start, c.end, snippet_chars)}
                    for n, c in enumerate(candidates, start=1)],
    )


@dataclass(frozen=True)
class LLMCall:
    output_text: str
    model_id: str | None
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: int


class LLMClassifier:
    def __init__(self, client: Any, config: LLMConfig, schema: dict[str, Any] | None = None):
        self._client = client
        if config.prompt_version not in PROMPTS:
            raise ExtractionConfigError([f"llm.prompt_version {config.prompt_version!r} is not one of {list(PROMPTS)}"])
        self.config = config
        self.instructions = PROMPTS[config.prompt_version]
        self.schema = schema or load_schema()

    @classmethod
    def from_config(cls, config: LLMConfig) -> LLMClassifier:
        from visibility_db.config import load_dotenv_file

        load_dotenv_file()
        key = os.environ.get(config.api_key_env)
        if not key:
            raise ExtractionConfigError([f"{config.api_key_env} is not set"])
        # max_retries=0: retries are decided by the caller, so every attempt is visible.
        return cls(openai.OpenAI(api_key=key, timeout=config.timeout_seconds, max_retries=0), config)

    @property
    def model(self) -> str:
        return self.config.model

    def request_params(self, request: ClassificationRequest) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "instructions": self.instructions,
            "input": request.payload(),
            "text": {"format": {"type": "json_schema", "name": self.schema["title"],
                                "schema": self.schema, "strict": True}},
            "reasoning": {"effort": self.config.reasoning_effort},
            "max_output_tokens": self.config.max_output_tokens,
            "store": False,
        }

    def check_access(self) -> str:
        """The model ID the API reports for the configured model. Raises ProviderError if unavailable."""
        try:
            return self._client.models.retrieve(self.config.model).id
        except openai.APIError as e:
            raise classify_error(e, {"model": self.config.model}, {}) from None

    def classify(self, request: ClassificationRequest) -> LLMCall:
        params = self.request_params(request)
        logged = {k: v for k, v in params.items() if k not in ("instructions", "input", "text")}
        logged["schema"] = self.schema["title"]
        started = time.monotonic()
        try:
            response = self._client.responses.create(**params)
        except openai.APIError as e:
            raise classify_error(e, logged, {}) from None
        latency_ms = int((time.monotonic() - started) * 1000)

        text = response.output_text or ""
        if response.status != "completed" or not text.strip():
            reason = (response.incomplete_details.reason if response.incomplete_details else None) or response.status
            raise ProviderError("incomplete_response", f"status {response.status!r} (reason: {reason})",
                                transient=False, model_id=response.model, request_params=logged)
        usage = response.usage
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        return LLMCall(
            output_text=text,
            model_id=response.model,
            input_tokens=usage.input_tokens if usage else None,
            cached_input_tokens=input_details.cached_tokens if input_details else None,
            output_tokens=usage.output_tokens if usage else None,
            reasoning_tokens=output_details.reasoning_tokens if output_details else None,
            latency_ms=latency_ms,
        )


# Models quote the words but drop markdown emphasis and may straighten typographic quotes.
_QUOTE_NOISE = str.maketrans({"*": None, "`": None, "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _normalized(text: str) -> str:
    return " ".join(text.translate(_QUOTE_NOISE).split())


def _check_ids(items: list[dict[str, Any]], key: str, expected: list[Any], errors: list[str]) -> None:
    got = [item[key] for item in items]
    duplicates = sorted({str(i) for i in got if got.count(i) > 1})
    missing = [i for i in expected if i not in got]
    unexpected = [i for i in got if i not in expected]
    if duplicates:
        errors.append(f"{key} returned more than once: {duplicates}")
    if missing:
        errors.append(f"{key} missing: {missing}")
    if unexpected:
        errors.append(f"{key} not in the request: {unexpected}")


def _check_evidence(quote: str, label: str, answer: str, errors: list[str], must_contain: str | None = None) -> None:
    if not quote.strip():
        errors.append(f"{label}: evidence is empty")
    elif len(quote) > MAX_EVIDENCE_CHARS:
        errors.append(f"{label}: evidence longer than {MAX_EVIDENCE_CHARS} characters")
    elif _normalized(quote) not in answer:
        errors.append(f"{label}: evidence is not a quote from the answer")
    elif must_contain and must_contain not in _normalized(quote):
        errors.append(f"{label}: evidence does not contain {must_contain!r}")


def validate_output(output_text: str, request: ClassificationRequest,
                    schema: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """(parsed output, []) if usable, otherwise (None, reasons)."""
    try:
        output = json.loads(output_text)
    except json.JSONDecodeError as e:
        return None, [f"output is not JSON: {e.msg}"]
    errors = [f"schema: {e.message} at /{'/'.join(map(str, e.absolute_path))}"
              for e in Draft202012Validator(schema).iter_errors(output)]
    if errors:
        return None, errors

    answer = _normalized(request.response_text)
    _check_ids(output["brands"], "brand_id", [b["brand_id"] for b in request.brands], errors)
    _check_ids(output["candidates"], "candidate_id", [c["candidate_id"] for c in request.candidates], errors)
    for b in output["brands"]:
        _check_evidence(b["evidence"], f"brand {b['brand_id']}", answer, errors)
    texts = {c["candidate_id"]: c["text"] for c in request.candidates}
    for c in output["candidates"]:
        _check_evidence(c["evidence"], f"candidate {c['candidate_id']}", answer, errors,
                        must_contain=texts.get(c["candidate_id"]))
    return (None, errors) if errors else (output, [])


def estimate_cost(price: Price | None, input_tokens: int | None, cached_input_tokens: int | None,
                  output_tokens: int | None) -> float | None:
    """USD, or None when the model has no configured price or usage is missing.

    OpenAI counts cached tokens inside input_tokens, and reasoning tokens inside output_tokens.
    """
    if price is None or input_tokens is None or output_tokens is None:
        return None
    cached = cached_input_tokens or 0
    cost = ((input_tokens - cached) * price.input + cached * price.cached_input
            + output_tokens * price.output) / 1_000_000
    return round(cost, 8)
