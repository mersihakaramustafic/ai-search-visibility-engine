"""Brand visibility rates per group of answers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any


METRICS = {
    "mention_rate": ("mentioned", "mentioned_answers"),
    "recommendation_rate": ("recommended", "recommended_answers"),
    "citation_rate": ("cited", "cited_answers"),
}
GROUP_FIELDS = ("provider", "intent")


@dataclass(frozen=True)
class Answer:
    observation_id: int
    query_id: str
    provider: str
    intent: str
    mentioned: frozenset[str]
    recommended: frozenset[str]  # recommendation other than 'none'
    cited: frozenset[str]  # at least one citation attributed to the brand's own site


def matched_queries(answers: Iterable[Answer]) -> list[Answer]:
    """Only answers to queries that every provider in the input answered."""
    answers = list(answers)
    providers: dict[str, set[str]] = defaultdict(set)
    for a in answers:
        providers[a.query_id].add(a.provider)
    every = {a.provider for a in answers}
    return [a for a in answers if providers[a.query_id] == every]


def brand_rates(answers: Iterable[Answer], brand_ids: Sequence[str], target_brand: str,
                by: Sequence[str] = GROUP_FIELDS) -> list[dict[str, Any]]:
    """One row per group and brand, with rates in 0..1 and the sample counts behind them."""
    if unknown := set(by) - set(GROUP_FIELDS):
        raise ValueError(f"cannot group by {sorted(unknown)}")
    groups: dict[tuple[str, ...], list[Answer]] = defaultdict(list)
    seen: set[int] = set()
    for a in answers:
        if a.observation_id in seen:
            raise ValueError(f"observation {a.observation_id} appears twice")
        seen.add(a.observation_id)
        groups[tuple(getattr(a, f) for f in by)].append(a)

    rows = []
    for key, group in sorted(groups.items()):
        runs: dict[str, list[Answer]] = defaultdict(list)
        for a in group:
            runs[a.query_id].append(a)
        for brand in brand_ids:
            row: dict[str, Any] = dict(zip(by, key))
            row.update(brand_id=brand, is_target_brand=brand == target_brand,
                       queries=len(runs), answers=len(group))
            for rate, (field, count) in METRICS.items():
                row[count] = sum(brand in getattr(a, field) for a in group)
                row[rate] = mean(mean(brand in getattr(a, field) for a in r) for r in runs.values())
            rows.append(row)
    return rows
