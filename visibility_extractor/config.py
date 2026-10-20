"""Load and validate configs/extraction.yaml."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "extraction.yaml"


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
class ExtractionConfig:
    matching: MatchingConfig
    citations: CitationConfig


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


def parse_config(raw: Any) -> ExtractionConfig:
    if not isinstance(raw, dict):
        raise ExtractionConfigError(["config must be a mapping"])
    errors: list[str] = []
    matching = parse_matching(raw.get("matching"), errors)
    citations = parse_citations(raw.get("citations"), errors)
    if errors:
        raise ExtractionConfigError(errors)
    return ExtractionConfig(matching=matching, citations=citations)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ExtractionConfig:
    if not path.is_file():
        raise ExtractionConfigError([f"{path} not found"])
    with path.open(encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f))
