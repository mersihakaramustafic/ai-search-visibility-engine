"""Deterministic priority score (0-100) for each validated recommendation.

Most of the score comes from measured gap data and the evidence a recommendation cites; the LLM's own
impact, confidence and effort ratings are a small, bounded part. All weights live in configs/recommendation.yaml.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, get_args

from visibility_analysis.gaps import COMPETITOR_GAPS, LABELS, PROVIDER_GAP, Opportunity
from visibility_recommender.evidence import EvidencePack
from visibility_recommender.output import Category, GenerationOutput, Recommendation

LEVELS = {"low": 0.0, "medium": 0.5, "high": 1.0}
EFFORT = {"low": 1.0, "medium": 0.5, "high": 0.0}  # less effort scores higher
OFFSITE = {"editorial_outreach", "community_visibility", "directory_listing"}  # act on other sites
SOURCE_KINDS = {"domain", "citation"}

PARTS = ("impact", "evidence", "relevance", "effort")
IMPACT_INPUTS = ("gap", "reach", "llm_impact")
EVIDENCE_INPUTS = ("coverage", "sample", "llm_confidence")
EVIDENCE_KINDS = ("citation", "domain", "query", "competitor_metric")
RELEVANCE_INPUTS = ("fit", "intent", "metric")
GAP_TYPES = (*COMPETITOR_GAPS, PROVIDER_GAP)
CATEGORIES = get_args(Category)
METRICS = tuple(LABELS)


@dataclass(frozen=True)
class ScoringConfig:
    version: str
    weights: dict[str, float]  # PARTS, sum 1
    impact: dict[str, float]  # IMPACT_INPUTS, sum 1
    full_gap: float  # a gap this large (rate difference) gets the full gap score
    evidence: dict[str, float]  # EVIDENCE_INPUTS, sum 1
    evidence_kinds: dict[str, float]  # EVIDENCE_KINDS, sum 1
    full_sample_queries: int
    relevance: dict[str, float]  # RELEVANCE_INPUTS, sum 1
    offsite_without_source: float  # fit multiplier for off-site actions citing no domain or citation
    fit: dict[str, dict[str, float]]  # gap_type -> category -> 0..1
    intent_values: dict[str, float]
    metric_values: dict[str, float]
    bands: dict[str, float]  # 'high' and 'medium' lower bounds on the 0-100 score


@dataclass(frozen=True)
class Score:
    opportunity_id: str
    index: int  # position in the LLM's list of recommendations
    scoring_version: str
    score: float  # 0-100, rounded to 2 decimals
    band: str  # 'high' | 'medium' | 'low'
    gap: float  # raw rate difference, used to break ties
    parts: dict[str, float]  # PARTS, each 0..1
    features: dict[str, float]  # every input, each 0..1, kept for later learning


def gap_size(o: Opportunity) -> float:
    other = o.compared_value if o.gap_type == PROVIDER_GAP else o.competitor_value
    return max(0.0, (other or 0.0) - o.target_value)


def _lookup(table: Mapping[str, float], key: str, what: str) -> float:
    if key not in table:
        raise ValueError(f"scoring has no {what} value for {key!r}; add it to configs/recommendation.yaml")
    return table[key]


def _cited_kinds(rec: Recommendation, pack: EvidencePack) -> set[str]:
    kinds = set()
    for evidence_id in rec.evidence_ids:
        if evidence_id not in pack.ids:
            continue  # the validator rejects these; ignore defensively
        item = pack.by_id(evidence_id)
        if item.kind == "metric":
            if item.data["brand_id"] != pack.opportunity.target_brand:
                kinds.add("competitor_metric")
        else:
            kinds.add(item.kind)
    return kinds


def _weighted(weights: Mapping[str, float], values: Mapping[str, float]) -> float:
    return sum(weights[k] * values[k] for k in weights)


def band(score: float, config: ScoringConfig) -> str:
    if score >= config.bands["high"]:
        return "high"
    return "medium" if score >= config.bands["medium"] else "low"


def score_recommendation(rec: Recommendation, index: int, pack: EvidencePack, config: ScoringConfig) -> Score:
    o = pack.opportunity
    gap = gap_size(o)
    kinds = _cited_kinds(rec, pack)

    fit = _lookup(config.fit.get(o.gap_type, {}), rec.category, f"fit ({o.gap_type})")
    if rec.category in OFFSITE and not kinds & SOURCE_KINDS:
        fit *= config.offsite_without_source

    features = {
        "gap": min(gap / config.full_gap, 1.0),
        "reach": min(len(o.supporting_query_ids) / max(o.queries, 1), 1.0),
        "llm_impact": LEVELS[rec.expected_impact],
        "coverage": sum(config.evidence_kinds[k] for k in kinds),
        "sample": min(o.queries / config.full_sample_queries, 1.0),
        "llm_confidence": LEVELS[rec.confidence],
        "fit": fit,
        "intent": _lookup(config.intent_values, o.intent, "intent"),
        "metric": _lookup(config.metric_values, o.metric, "metric"),
        "effort": EFFORT[rec.effort],
    }
    parts = {
        "impact": _weighted(config.impact, features),
        "evidence": _weighted(config.evidence, features),
        "relevance": _weighted(config.relevance, features),
        "effort": features["effort"],
    }
    score = round(100 * _weighted(config.weights, parts), 2)
    return Score(pack.opportunity_id, index, config.version, score, band(score, config), gap, parts, features)


def score_output(output: GenerationOutput, pack: EvidencePack, config: ScoringConfig) -> list[Score]:
    return [score_recommendation(rec, i, pack, config) for i, rec in enumerate(output.recommendations)]


def rank(scores: Iterable[Score]) -> list[Score]:
    """Highest score first; ties go to the larger gap, then opportunity id, then the LLM's order."""
    return sorted(scores, key=lambda s: (-s.score, -s.gap, s.opportunity_id, s.index))


# --- config parsing (called from visibility_recommender.config) ---

def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _unit_map(raw: Any, path: str, names: Iterable[str] | None, errors: list[str]) -> dict[str, float]:
    """A mapping of names to numbers from 0 to 1; exactly `names` when given."""
    if not isinstance(raw, dict) or not raw:
        errors.append(f"{path} must be a non-empty mapping")
        return {}
    if names is not None and set(raw) != set(names):
        errors.append(f"{path} must have exactly these keys: {', '.join(names)}")
        return {}
    if not all(_number(v) and 0 <= v <= 1 for v in raw.values()):
        errors.append(f"{path} values must be numbers from 0 to 1")
        return {}
    return {str(k): float(v) for k, v in raw.items()}


def _weights(raw: Any, path: str, names: Iterable[str], errors: list[str]) -> dict[str, float]:
    values = _unit_map(raw, path, names, errors)
    if values and not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
        errors.append(f"{path} must add up to 1 (got {sum(values.values()):g})")
        return {}
    return values


def _section(raw: Mapping[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        errors.append(f"scoring.{key} must be a mapping")
        return {}
    return value


def parse_scoring(raw: Any, errors: list[str], version_re: Any) -> ScoringConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        errors.append("scoring must be a mapping")
        return None
    start = len(errors)
    version = raw.get("version")
    if not isinstance(version, str) or not version_re.match(version):
        errors.append("scoring.version must be a short label like 'p1'")
    impact, evidence, relevance = (_section(raw, k, errors) for k in ("impact", "evidence", "relevance"))

    full_gap = impact.get("full_gap")
    if not _number(full_gap) or not 0 < full_gap <= 1:
        errors.append("scoring.impact.full_gap must be a number above 0 and at most 1")
    full_sample = evidence.get("full_sample_queries")
    if not isinstance(full_sample, int) or isinstance(full_sample, bool) or full_sample < 1:
        errors.append("scoring.evidence.full_sample_queries must be a positive integer")
    offsite = relevance.get("offsite_without_source")
    if not _number(offsite) or not 0 <= offsite <= 1:
        errors.append("scoring.relevance.offsite_without_source must be a number from 0 to 1")

    fit_raw = relevance.get("fit")
    fit: dict[str, dict[str, float]] = {}
    if not isinstance(fit_raw, dict) or set(fit_raw) != set(GAP_TYPES):
        errors.append(f"scoring.relevance.fit must have exactly these gap types: {', '.join(GAP_TYPES)}")
    else:
        fit = {g: _unit_map(fit_raw[g], f"scoring.relevance.fit.{g}", CATEGORIES, errors) for g in GAP_TYPES}

    bands = raw.get("bands")
    if (not isinstance(bands, dict) or set(bands) != {"high", "medium"}
            or not all(_number(v) for v in bands.values()) or not 0 <= bands["medium"] <= bands["high"] <= 100):
        errors.append("scoring.bands needs 'high' and 'medium' with 0 <= medium <= high <= 100")
        bands = {}

    config = ScoringConfig(
        version=version,
        weights=_weights(raw.get("weights"), "scoring.weights", PARTS, errors),
        impact=_weights(impact.get("weights"), "scoring.impact.weights", IMPACT_INPUTS, errors),
        full_gap=full_gap,
        evidence=_weights(evidence.get("weights"), "scoring.evidence.weights", EVIDENCE_INPUTS, errors),
        evidence_kinds=_weights(evidence.get("kinds"), "scoring.evidence.kinds", EVIDENCE_KINDS, errors),
        full_sample_queries=full_sample,
        relevance=_weights(relevance.get("weights"), "scoring.relevance.weights", RELEVANCE_INPUTS, errors),
        offsite_without_source=offsite,
        fit=fit,
        intent_values=_unit_map(relevance.get("intent_values"), "scoring.relevance.intent_values", None, errors),
        metric_values=_unit_map(relevance.get("metric_values"), "scoring.relevance.metric_values", METRICS, errors),
        bands={k: float(v) for k, v in bands.items()},
    )
    return config if len(errors) == start else None
