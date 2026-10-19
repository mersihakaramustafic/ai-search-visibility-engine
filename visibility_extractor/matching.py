"""Deterministic brand matching: find which tracked brands an answer mentions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from visibility_extractor.config import MatchingConfig

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


@dataclass(frozen=True)
class Brand:
    brand_id: str
    brand_name: str
    canonical_domain: str
    aliases: tuple[str, ...]
    is_primary_brand: bool


@dataclass(frozen=True)
class Occurrence:
    brand_id: str
    start: int
    end: int
    text: str
    ambiguous: bool


@dataclass(frozen=True)
class MatchResult:
    confirmed: list[Occurrence]
    candidates: list[Occurrence]
    candidates_dropped: int  # ambiguous occurrences over max_candidates_per_brand


@dataclass(frozen=True)
class MentionSummary:
    brand_id: str
    mention_count: int
    first_char_offset: int
    first_mention_rank: int
    matched_alias: str
    context_snippet: str
    match_method: str  # 'alias' | 'llm_disambiguated'


def _whole_word(form: str, case_sensitive: bool) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(form) + r"(?!\w)", 0 if case_sensitive else re.IGNORECASE)


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def snippet(text: str, start: int, end: int, width: int) -> str:
    return " ".join(text[max(0, start - width):end + width].split())


class BrandMatcher:
    def __init__(self, brands: list[Brand], config: MatchingConfig):
        self.config = config
        self.exclusions = [re.compile(e) for e in config.exclusions]
        known = {b.brand_id: b for b in brands}

        errors = []
        for brand_id in set(config.ambiguous_aliases) | set(config.extra_ambiguous_forms):
            if brand_id not in known:
                errors.append(f"unknown brand_id {brand_id!r} in matching config")
        for brand_id, forms in config.ambiguous_aliases.items():
            if brand_id in known:
                errors += [f"{brand_id}: ambiguous alias {f!r} is not one of its aliases"
                           for f in forms if f not in known[brand_id].aliases]
        if errors:
            raise ValueError("; ".join(errors))

        # (brand_id, compiled pattern, ambiguous?) for every form we look for.
        self._patterns: list[tuple[str, re.Pattern[str], bool]] = []
        for b in brands:
            ambiguous = set(config.ambiguous_aliases.get(b.brand_id, ()))
            seen: set[str] = set()
            for alias in b.aliases:
                if alias not in ambiguous and alias.lower() not in seen:  # "HubSpot"/"Hubspot": one pattern
                    seen.add(alias.lower())
                    self._patterns.append((b.brand_id, _whole_word(alias, case_sensitive=False), False))
            for form in (*sorted(ambiguous), *config.extra_ambiguous_forms.get(b.brand_id, ())):
                self._patterns.append((b.brand_id, _whole_word(form, case_sensitive=True), True))

    def masked_spans(self, text: str, extra: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
        spans = [m.span() for m in URL_RE.finditer(text)]
        for pattern in self.exclusions:
            spans += [m.span() for m in pattern.finditer(text)]
        return spans + list(extra or [])

    def match(self, text: str, extra_masks: list[tuple[int, int]] | None = None) -> MatchResult:
        # 1. Mask
        masked = self.masked_spans(text, extra_masks)

        # 2. Find
        found = [
            Occurrence(brand_id, m.start(), m.end(), m.group(0), ambiguous)
            for brand_id, pattern, ambiguous in self._patterns
            for m in pattern.finditer(text)
            if not _overlaps(m.start(), m.end(), masked)
        ]

        # 3. Resolve: by position, longest first; on equal length a normal alias beats an ambiguous one.
        found.sort(key=lambda o: (o.start, -(o.end - o.start), o.ambiguous))
        kept: list[Occurrence] = []
        for o in found:
            if not kept or o.start >= kept[-1].end:
                kept.append(o)

        # 4. Split, capping candidates per brand.
        confirmed = [o for o in kept if not o.ambiguous]
        candidates: list[Occurrence] = []
        per_brand: dict[str, int] = {}
        dropped = 0
        for o in kept:
            if not o.ambiguous:
                continue
            per_brand[o.brand_id] = per_brand.get(o.brand_id, 0) + 1
            if per_brand[o.brand_id] > self.config.max_candidates_per_brand:
                dropped += 1
            else:
                candidates.append(o)
        return MatchResult(confirmed, candidates, dropped)


def aggregate_mentions(text: str, occurrences: list[Occurrence], snippet_chars: int) -> list[MentionSummary]:
    """One summary per brand, ranked by where each brand first appears (1 = first)."""
    by_brand: dict[str, list[Occurrence]] = {}
    for o in sorted(occurrences, key=lambda o: o.start):
        by_brand.setdefault(o.brand_id, []).append(o)

    summaries = []
    ranked = sorted(by_brand.items(), key=lambda item: item[1][0].start)
    for rank, (brand_id, occs) in enumerate(ranked, start=1):
        first = occs[0]
        summaries.append(MentionSummary(
            brand_id=brand_id,
            mention_count=len(occs),
            first_char_offset=first.start,
            first_mention_rank=rank,
            matched_alias=first.text,
            context_snippet=snippet(text, first.start, first.end, snippet_chars),
            # 'alias' if at least one occurrence needed no confirmation.
            match_method="alias" if any(not o.ambiguous for o in occs) else "llm_disambiguated",
        ))
    return summaries
