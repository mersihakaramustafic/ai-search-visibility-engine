"""Settings for recommendation generation, loaded from configs/recommendation.yaml."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from visibility_collector.config import RetryConfig
from visibility_extractor.config import VERSION_RE, Price, parse_pricing, parse_retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "recommendation.yaml"
MODEL_ENV_VAR = "RECOMMENDATION_MODEL"
PROVIDERS = ("gemini",)


class RecommendationConfigError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class GeneratorConfig:
    provider: str
    model: str
    api_key_env: str
    timeout_seconds: int
    max_output_tokens: int
    temperature: float
    prompt_version: str
    schema_version: str


@dataclass(frozen=True)
class RecommendationConfig:
    generator_version: str
    llm: GeneratorConfig
    retry: RetryConfig = field(default_factory=RetryConfig)
    pricing: dict[str, Price] = field(default_factory=dict)

    @property
    def price(self) -> Price | None:
        return self.pricing.get(self.llm.model)


def _label(raw: Mapping[str, Any], key: str, errors: list[str], value: Any = None) -> str:
    value = value or raw.get(key)
    if not isinstance(value, str) or not VERSION_RE.match(value):
        errors.append(f"llm.{key} must be a short label (letters, digits, '.', '_', '-')")
        return ""
    return value


def _positive_int(raw: Mapping[str, Any], key: str, errors: list[str]) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        errors.append(f"llm.{key} must be a positive integer")
        return 0
    return value


def parse_config(raw: Any, env: Mapping[str, str] | None = None) -> RecommendationConfig:
    if not isinstance(raw, dict):
        raise RecommendationConfigError(["config must be a mapping"])
    errors: list[str] = []
    version = raw.get("generator_version")
    if not isinstance(version, str) or not VERSION_RE.match(version):
        errors.append("generator_version must be a short label like 'recs-v1'")
    llm = raw.get("llm")
    if not isinstance(llm, dict):
        raise RecommendationConfigError(errors + ["llm must be a mapping"])
    if llm.get("provider") not in PROVIDERS:
        errors.append(f"llm.provider must be one of {list(PROVIDERS)}")
    if not isinstance(llm.get("api_key_env"), str) or not llm["api_key_env"]:
        errors.append("llm.api_key_env must name an environment variable")
    temperature = llm.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
        errors.append("llm.temperature must be a number from 0 to 2")
    generator = GeneratorConfig(
        provider=llm.get("provider"),
        model=_label(llm, "model", errors, (env or {}).get(MODEL_ENV_VAR)),
        api_key_env=llm.get("api_key_env"),
        timeout_seconds=_positive_int(llm, "timeout_seconds", errors),
        max_output_tokens=_positive_int(llm, "max_output_tokens", errors),
        temperature=float(temperature) if isinstance(temperature, (int, float)) else 0.0,
        prompt_version=_label(llm, "prompt_version", errors),
        schema_version=_label(llm, "schema_version", errors),
    )
    retry = parse_retry(raw.get("retry"), errors)
    pricing = parse_pricing(raw.get("pricing"), errors)
    if errors:
        raise RecommendationConfigError(errors)
    return RecommendationConfig(version, generator, retry, pricing)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> RecommendationConfig:
    if not path.is_file():
        raise RecommendationConfigError([f"{path} not found"])
    from visibility_db.config import load_dotenv_file

    load_dotenv_file()  # RECOMMENDATION_MODEL may be set in .env
    return parse_config(yaml.safe_load(path.read_text(encoding="utf-8")), os.environ)
