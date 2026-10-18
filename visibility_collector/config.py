"""Load and validate configs/collection.yaml.

Every problem is collected and reported at once, so a broken config fails
before any database row or API call is touched.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "collection.yaml"
SUPPORTED_PROVIDERS = ("openai", "gemini")
PILOT_MIN_QUERIES = 3
PILOT_MAX_QUERIES = 5
QUERY_ID_RE = re.compile(r"^Q[0-9]{3,}$")
ENV_VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class CollectorConfigError(Exception):
    """Raised when the collection config is invalid."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid collection config:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    api_key_env: str
    timeout_seconds: float
    max_output_tokens: int
    concurrency: int
    options: dict[str, Any] = field(default_factory=dict)

    def api_key(self) -> str:
        """Read the key at call time; it is never stored on the config object."""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise CollectorConfigError([f"{self.name}: environment variable {self.api_key_env} is not set"])
        return key


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 4
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 60.0


@dataclass(frozen=True)
class CollectionConfig:
    providers: dict[str, ProviderConfig]
    retry: RetryConfig
    pilot_query_ids: tuple[str, ...]


def validate_pilot_query_ids(query_ids: list[str] | tuple[str, ...]) -> list[str]:
    errors = []
    if not PILOT_MIN_QUERIES <= len(query_ids) <= PILOT_MAX_QUERIES:
        errors.append(f"pilot needs {PILOT_MIN_QUERIES}-{PILOT_MAX_QUERIES} query IDs, got {len(query_ids)}")
    if len(set(query_ids)) != len(query_ids):
        errors.append("pilot query IDs contain duplicates")
    bad = [q for q in query_ids if not isinstance(q, str) or not QUERY_ID_RE.match(q)]
    if bad:
        errors.append(f"pilot query IDs must look like Q001: {bad}")
    return errors


def _positive(section: dict, key: str, label: str, errors: list[str], kind: type = int) -> Any:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (kind is int and not isinstance(value, int)):
        errors.append(f"{label}.{key} must be a positive {'integer' if kind is int else 'number'}")
        return None
    if value <= 0:
        errors.append(f"{label}.{key} must be greater than 0")
        return None
    return value


def parse_config(raw: Any) -> CollectionConfig:
    errors: list[str] = []
    if not isinstance(raw, dict):
        raise CollectorConfigError(["config must be a mapping"])

    providers: dict[str, ProviderConfig] = {}
    raw_providers = raw.get("providers")
    if not isinstance(raw_providers, dict) or not raw_providers:
        errors.append("providers must be a non-empty mapping")
        raw_providers = {}
    for name, section in raw_providers.items():
        label = f"providers.{name}"
        if name not in SUPPORTED_PROVIDERS:
            errors.append(f"{label}: unsupported provider (supported: {', '.join(SUPPORTED_PROVIDERS)})")
            continue
        if not isinstance(section, dict):
            errors.append(f"{label} must be a mapping")
            continue
        model = section.get("model")
        if not isinstance(model, str) or not model.strip():
            errors.append(f"{label}.model must be a non-empty string")
        api_key_env = section.get("api_key_env")
        if not isinstance(api_key_env, str) or not ENV_VAR_RE.match(api_key_env):
            errors.append(f"{label}.api_key_env must name an environment variable, e.g. OPENAI_API_KEY")
        timeout = _positive(section, "timeout_seconds", label, errors, kind=float)
        max_tokens = _positive(section, "max_output_tokens", label, errors)
        concurrency = _positive(section, "concurrency", label, errors)
        known = {"model", "api_key_env", "timeout_seconds", "max_output_tokens", "concurrency"}
        options = {k: v for k, v in section.items() if k not in known}
        if name == "openai":
            web_search = options.get("web_search", {})
            if not isinstance(web_search, dict) or not isinstance(web_search.get("use_market_location", False), bool):
                errors.append(f"{label}.web_search.use_market_location must be true or false")
        unknown = set(options) - {"web_search"}
        if unknown:
            errors.append(f"{label}: unknown keys {sorted(unknown)}")
        if not errors:
            providers[name] = ProviderConfig(
                name=name,
                model=model.strip(),
                api_key_env=api_key_env,
                timeout_seconds=float(timeout),
                max_output_tokens=max_tokens,
                concurrency=concurrency,
                options=options,
            )

    raw_retry = raw.get("retry", {})
    retry = RetryConfig()
    if not isinstance(raw_retry, dict):
        errors.append("retry must be a mapping")
    elif raw_retry:
        attempts = _positive(raw_retry, "max_attempts", "retry", errors)
        initial = _positive(raw_retry, "initial_backoff_seconds", "retry", errors, kind=float)
        maximum = _positive(raw_retry, "max_backoff_seconds", "retry", errors, kind=float)
        if initial and maximum and initial > maximum:
            errors.append("retry.initial_backoff_seconds must not exceed retry.max_backoff_seconds")
        if attempts and initial and maximum:
            retry = RetryConfig(attempts, float(initial), float(maximum))

    pilot = raw.get("pilot", {})
    query_ids = pilot.get("query_ids") if isinstance(pilot, dict) else None
    if not isinstance(query_ids, list):
        errors.append("pilot.query_ids must be a list")
        query_ids = []
    else:
        errors.extend(validate_pilot_query_ids(query_ids))

    if errors:
        raise CollectorConfigError(errors)
    return CollectionConfig(providers=providers, retry=retry, pilot_query_ids=tuple(query_ids))


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> CollectionConfig:
    if not path.is_file():
        raise CollectorConfigError([f"{path} not found"])
    with path.open(encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f))
