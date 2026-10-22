"""Load and validate configs/extraction.yaml."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from visibility_collector.config import RetryConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "extraction.yaml"
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MODEL_ENV_VAR = "EXTRACTION_MODEL"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")
STRATEGIES = ("rules_only", "rules_and_llm")


class ExtractionConfigError(Exception):

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid extraction config:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass(frozen=True)
class MatchingConfig:
    ambiguous_aliases: dict[str, tuple[str, ...]]
    extra_ambiguous_forms: dict[str, tuple[str, ...]]
    exclusions: tuple[str, ...]
    max_candidates_per_brand: int
    snippet_chars: int


@dataclass(frozen=True)
class CitationConfig:
    # brand_id -> substrings; a citation on that brand's domain counts as the brand's
    # only if its lowercased "host/path" contains one of them.
    brand_url_rules: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key_env: str
    timeout_seconds: int
    max_output_tokens: int
    reasoning_effort: str
    prompt_version: str
    schema_version: str
    concurrency: int = 1


@dataclass(frozen=True)
class Price:
    input: float
    cached_input: float
    output: float


@dataclass(frozen=True)
class ExtractionConfig:
    extractor_version: str
    matching: MatchingConfig
    citations: CitationConfig
    llm: LLMConfig | None = None
    pricing: dict[str, Price] = field(default_factory=dict)
    # source provider -> 'rules_only' | 'rules_and_llm'; providers not listed are rules_only
    strategy: dict[str, str] = field(default_factory=dict)
    retry: RetryConfig = field(default_factory=RetryConfig)
    pilot_per_provider: int = 8

    def strategy_for(self, provider: str) -> str:
        return self.strategy.get(provider, "rules_only")


def _brand_forms(value: Any, label: str, errors: list[str]) -> dict[str, tuple[str, ...]]:
    """A mapping of brand_id -> non-empty list of non-empty strings."""
    if value is None:
        return {}
    ok = isinstance(value, dict) and all(
        isinstance(forms, list) and forms and all(isinstance(f, str) and f.strip() for f in forms)
        for forms in value.values()
    )
    if not ok:
        errors.append(f"{label} must map brand IDs to non-empty lists of strings")
        return {}
    return {brand_id: tuple(forms) for brand_id, forms in value.items()}


def _positive_int(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        errors.append(f"{label} must be a positive integer")
        return 0
    return value


def parse_matching(raw: Any, errors: list[str]) -> MatchingConfig:
    if not isinstance(raw, dict):
        errors.append("matching must be a mapping")
        raw = {}

    exclusions = raw.get("exclusions") or []
    if not isinstance(exclusions, list) or not all(isinstance(e, str) and e for e in exclusions):
        errors.append("matching.exclusions must be a list of regex strings")
        exclusions = []
    for pattern in exclusions:
        try:
            re.compile(pattern)
        except re.error as e:
            errors.append(f"matching.exclusions: invalid regex {pattern!r}: {e}")

    return MatchingConfig(
        ambiguous_aliases=_brand_forms(raw.get("ambiguous_aliases"), "matching.ambiguous_aliases", errors),
        extra_ambiguous_forms=_brand_forms(raw.get("extra_ambiguous_forms"), "matching.extra_ambiguous_forms", errors),
        exclusions=tuple(exclusions),
        max_candidates_per_brand=_positive_int(raw.get("max_candidates_per_brand"),
                                               "matching.max_candidates_per_brand", errors),
        snippet_chars=_positive_int(raw.get("snippet_chars"), "matching.snippet_chars", errors),
    )


def parse_citations(raw: Any, errors: list[str]) -> CitationConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        errors.append("citations must be a mapping")
        raw = {}
    rules = _brand_forms(raw.get("brand_url_rules"), "citations.brand_url_rules", errors)
    return CitationConfig(brand_url_rules={b: tuple(r.lower() for r in rs) for b, rs in rules.items()})


def _label(value: Any, label: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not VERSION_RE.match(value):
        errors.append(f"{label} must be a short label (letters, digits, '.', '_', '-')")
        return ""
    return value


def parse_llm(raw: Any, env: Mapping[str, str], errors: list[str]) -> LLMConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        errors.append("llm must be a mapping")
        return None
    effort = raw.get("reasoning_effort")
    if effort not in REASONING_EFFORTS:
        errors.append(f"llm.reasoning_effort must be one of {list(REASONING_EFFORTS)}")
    api_key_env = raw.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env:
        errors.append("llm.api_key_env must name an environment variable")
    return LLMConfig(
        model=_label(env.get(MODEL_ENV_VAR) or raw.get("model"), "llm.model", errors),
        api_key_env=api_key_env,
        timeout_seconds=_positive_int(raw.get("timeout_seconds"), "llm.timeout_seconds", errors),
        max_output_tokens=_positive_int(raw.get("max_output_tokens"), "llm.max_output_tokens", errors),
        reasoning_effort=effort,
        prompt_version=_label(raw.get("prompt_version"), "llm.prompt_version", errors),
        schema_version=_label(raw.get("schema_version"), "llm.schema_version", errors),
        concurrency=_positive_int(raw.get("concurrency", 1), "llm.concurrency", errors),
    )


def parse_strategy(raw: Any, has_llm: bool, errors: list[str]) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not all(v in STRATEGIES for v in raw.values()):
        errors.append(f"strategy must map providers to one of {list(STRATEGIES)}")
        return {}
    if not has_llm and "rules_and_llm" in raw.values():
        errors.append("strategy rules_and_llm needs an llm section")
    return dict(raw)


def parse_retry(raw: Any, errors: list[str]) -> RetryConfig:
    if raw is None:
        return RetryConfig()
    if not isinstance(raw, dict):
        errors.append("retry must be a mapping")
        return RetryConfig()
    numbers_ok = all(isinstance(raw.get(k), (int, float)) and not isinstance(raw.get(k), bool) and raw[k] > 0
                     for k in ("initial_backoff_seconds", "max_backoff_seconds"))
    if not numbers_ok:
        errors.append("retry backoff seconds must be positive numbers")
        return RetryConfig()
    return RetryConfig(max_attempts=_positive_int(raw.get("max_attempts"), "retry.max_attempts", errors),
                       initial_backoff_seconds=float(raw["initial_backoff_seconds"]),
                       max_backoff_seconds=float(raw["max_backoff_seconds"]))


def parse_pricing(raw: Any, errors: list[str]) -> dict[str, Price]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        errors.append("pricing must map model names to prices")
        return {}
    prices = {}
    for model, p in raw.items():
        keys = ("input", "cached_input", "output")
        if not isinstance(p, dict) or set(p) != set(keys) or not all(
                isinstance(p[k], (int, float)) and not isinstance(p[k], bool) and p[k] >= 0 for k in keys):
            errors.append(f"pricing.{model} needs non-negative input, cached_input and output")
            continue
        prices[model] = Price(**{k: float(p[k]) for k in keys})
    return prices


def parse_config(raw: Any, env: Mapping[str, str] | None = None) -> ExtractionConfig:
    if not isinstance(raw, dict):
        raise ExtractionConfigError(["config must be a mapping"])
    errors: list[str] = []
    version = raw.get("extractor_version")
    if not isinstance(version, str) or not VERSION_RE.match(version):
        errors.append("extractor_version must be a short label like 'rules-v1' (letters, digits, '.', '_', '-')")
    matching = parse_matching(raw.get("matching"), errors)
    citations = parse_citations(raw.get("citations"), errors)
    llm = parse_llm(raw.get("llm"), env or {}, errors)
    pricing = parse_pricing(raw.get("pricing"), errors)
    strategy = parse_strategy(raw.get("strategy"), llm is not None, errors)
    retry = parse_retry(raw.get("retry"), errors)
    pilot = raw.get("pilot") or {}
    pilot_per_provider = _positive_int(pilot.get("observations_per_provider", 8) if isinstance(pilot, dict) else None,
                                       "pilot.observations_per_provider", errors)
    if errors:
        raise ExtractionConfigError(errors)
    return ExtractionConfig(extractor_version=version, matching=matching, citations=citations, llm=llm,
                            pricing=pricing, strategy=strategy, retry=retry, pilot_per_provider=pilot_per_provider)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ExtractionConfig:
    if not path.is_file():
        raise ExtractionConfigError([f"{path} not found"])
    from visibility_db.config import load_dotenv_file

    load_dotenv_file()  # EXTRACTION_MODEL may be set in .env
    with path.open(encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f), os.environ)
