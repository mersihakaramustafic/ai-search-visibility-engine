"""Rules-only extraction of one observation: citations, then brand mentions, then validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from visibility_extractor.citations import Citation, DomainAttributor, extract_citations
from visibility_extractor.config import ExtractionConfig
from visibility_extractor.matching import Brand, BrandMatcher, MentionSummary, aggregate_mentions


@dataclass(frozen=True)
class Observation:
    observation_id: int
    provider: str
    response_text: str
    grounding_metadata: Any


@dataclass(frozen=True)
class Extraction:
    observation_id: int
    extractor_version: str
    status: str  # 'succeeded' | 'failed'
    validation_status: str | None  # 'valid' | 'valid_with_warnings' | 'invalid'; None when failed
    validation_issues: list[str] = field(default_factory=list)
    target_brand_mentioned: bool | None = None
    competitor_brands_mentioned: int | None = None
    candidate_count: int | None = None
    mentions: list[MentionSummary] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


def validate(text: str, mentions: list[MentionSummary], citations: list[Citation],
             brand_ids: set[str]) -> list[str]:
    """Errors that make the extraction unusable. Empty means it can be stored."""
    errors = []
    for m in mentions:
        if m.brand_id not in brand_ids:
            errors.append(f"mention of unknown brand {m.brand_id!r}")
        found = text[m.first_char_offset:m.first_char_offset + len(m.matched_alias)]
        if found != m.matched_alias:
            errors.append(f"{m.brand_id}: text at offset {m.first_char_offset} is {found!r}, "
                          f"not {m.matched_alias!r}")
        if m.mention_count < 1:
            errors.append(f"{m.brand_id}: mention_count {m.mention_count}")
    if sorted(m.first_mention_rank for m in mentions) != list(range(1, len(mentions) + 1)):
        errors.append("mention ranks are not 1..n")
    if [c.citation_position for c in citations] != list(range(1, len(citations) + 1)):
        errors.append("citation positions are not 1..n")
    for c in citations:
        if c.brand_id is not None and c.brand_id not in brand_ids:
            errors.append(f"citation {c.citation_position} names unknown brand {c.brand_id!r}")
    return errors


class RulesExtractor:
    def __init__(self, brands: list[Brand], config: ExtractionConfig):
        self.version = config.extractor_version
        self.snippet_chars = config.matching.snippet_chars
        self.matcher = BrandMatcher(brands, config.matching)
        self.attributor = DomainAttributor(brands, config.citations)
        self.brand_ids = {b.brand_id for b in brands}
        self.target_ids = {b.brand_id for b in brands if b.is_primary_brand}

    def extract(self, obs: Observation) -> Extraction:
        try:
            return self._extract(obs)
        except Exception as e:  # one bad observation must not stop the run
            return Extraction(obs.observation_id, self.version, "failed", None,
                              error_type=type(e).__name__, error_message=str(e)[:500] or type(e).__name__)

    def _extract(self, obs: Observation) -> Extraction:
        text = obs.response_text
        cites = extract_citations(obs.provider, text, obs.grounding_metadata, self.attributor)
        match = self.matcher.match(text, cites.masked_spans)
        mentions = aggregate_mentions(text, match.confirmed, self.snippet_chars)

        warnings = list(cites.warnings)
        if match.candidates_dropped:
            warnings.append(f"{match.candidates_dropped} ambiguous occurrence(s) over the per-brand cap dropped")
        errors = validate(text, mentions, cites.citations, self.brand_ids)
        if errors:
            # Nothing is stored for an invalid result; the issues say why.
            return Extraction(obs.observation_id, self.version, "succeeded", "invalid", errors + warnings)

        mentioned = {m.brand_id for m in mentions}
        return Extraction(
            observation_id=obs.observation_id,
            extractor_version=self.version,
            status="succeeded",
            validation_status="valid_with_warnings" if warnings else "valid",
            validation_issues=warnings,
            target_brand_mentioned=bool(mentioned & self.target_ids),
            competitor_brands_mentioned=len(mentioned - self.target_ids),
            candidate_count=len(match.candidates),
            mentions=mentions,
            citations=cites.citations,
        )
