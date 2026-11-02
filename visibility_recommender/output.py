"""The structured output the LLM must return, and checks that it uses only the evidence it was given."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from visibility_recommender.evidence import EvidencePack, pct

Level = Literal["low", "medium", "high"]
# The action types recommendation_candidates already allows.
Category = Literal["owned_content_optimization", "editorial_outreach", "community_visibility", "directory_listing"]
MAX_RECOMMENDATIONS = 3

PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
URL_RE = re.compile(r"https?://[^\s)\"'>\]]+")
DOMAIN_RE = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:com|org|net|io|ai|app|co|me|agency|de|uk)\b",
                       re.IGNORECASE)
GUARANTEE_RE = re.compile(r"\bguarantee[sd]?\b|\bwill (?:increase|improve|boost|raise)\b", re.IGNORECASE)


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    title: str = Field(min_length=5, max_length=120)
    action: str = Field(min_length=20, max_length=1000)
    rationale: str = Field(min_length=20, max_length=800)
    evidence_ids: list[str] = Field(min_length=1)
    expected_impact: Level
    effort: Level
    confidence: Level


class GenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opportunity_id: str
    insufficient_evidence: bool
    insufficient_reason: str | None = None
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=MAX_RECOMMENDATIONS)

    @model_validator(mode="after")
    def _consistent(self) -> GenerationOutput:
        if self.insufficient_evidence and (self.recommendations or not self.insufficient_reason):
            raise ValueError("insufficient_evidence needs a reason and no recommendations")
        if not self.insufficient_evidence and not self.recommendations:
            raise ValueError("at least one recommendation is required unless evidence is insufficient")
        return self


def allowed_percentages(pack: EvidencePack) -> set[str]:
    values = {pct(i.data["value"]) for i in pack.items if i.kind == "metric"}
    o = pack.opportunity
    if o.competitor_value is not None:
        values.add(pct(abs(o.competitor_value - o.target_value)))
    if o.compared_value is not None:
        values.add(pct(abs(o.compared_value - o.target_value)))
    return values


def allowed_domains(pack: EvidencePack, brand_domains: set[str]) -> set[str]:
    return brand_domains | {i.data["domain"] for i in pack.items if i.kind in ("domain", "citation")}


def _known_domain(found: str, allowed: set[str]) -> bool:
    found = found.lower().removeprefix("www.")
    return any(found == d or found.endswith("." + d) for d in allowed)


def check_recommendation(r: Recommendation, pack: EvidencePack, brand_domains: set[str]) -> list[str]:
    errors = []
    unknown = sorted(set(r.evidence_ids) - pack.ids)
    if unknown:
        errors.append(f"unknown evidence ids {unknown}")
    if not any(pack.by_id(e).kind != "metric" for e in r.evidence_ids if e in pack.ids):
        errors.append("cites only metrics; at least one query, domain or citation is required")
    text = " ".join((r.title, r.action, r.rationale))
    percentages = allowed_percentages(pack)
    for p in PERCENT_RE.findall(text):
        if pct(float(p.rstrip("% ")) / 100) not in percentages:
            errors.append(f"percentage {p.strip()!r} is not in the evidence")
    urls = {i.data["url"] for i in pack.items if i.kind == "citation"}
    for u in URL_RE.findall(text):
        if u.rstrip(".,;:") not in urls:
            errors.append(f"URL {u!r} is not in the evidence")
    domains = allowed_domains(pack, brand_domains)
    for d in DOMAIN_RE.findall(URL_RE.sub(" ", text)):
        if not _known_domain(d, domains):
            errors.append(f"domain {d!r} is not in the evidence")
    if m := GUARANTEE_RE.search(text):
        errors.append(f"claims a certain outcome: {m.group(0)!r}")
    return errors


def validate_output(output_text: str, pack: EvidencePack,
                    brand_domains: set[str]) -> tuple[GenerationOutput | None, list[str]]:
    """The parsed output and an empty list, or None and every problem found."""
    try:
        output = GenerationOutput.model_validate_json(output_text)
    except ValidationError as e:
        return None, [f"{'.'.join(map(str, err['loc'])) or 'output'}: {err['msg']}" for err in e.errors()]
    errors = []
    if output.opportunity_id != pack.opportunity_id:
        errors.append(f"opportunity_id {output.opportunity_id!r} does not match {pack.opportunity_id!r}")
    for n, r in enumerate(output.recommendations, 1):
        errors += [f"recommendation {n}: {e}" for e in check_recommendation(r, pack, brand_domains)]
    return (None, errors) if errors else (output, [])
