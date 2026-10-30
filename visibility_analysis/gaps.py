"""Visibility gaps of a target brand, found with fixed rules on the brand_rates metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean

from visibility_analysis.metrics import METRICS, Answer, brand_rates, matched_queries

COMPETITOR_GAPS = {
    "competitor_recommended_more": "recommendation_rate",
    "competitor_cited_more": "citation_rate",
    "target_mentioned_less": "mention_rate",
}
PROVIDER_GAP = "weaker_on_provider"

LABELS = {"mention_rate": "mention rate", "recommendation_rate": "recommendation rate",
          "citation_rate": "citation rate"}


@dataclass(frozen=True)
class GapRules:
    min_queries: int = 10  # per provider and intent
    min_gap: float = 0.10  # difference in rate, 0..1
    min_queries_ahead: int = 3  # queries where the gap is visible
    min_domain_queries: int = 3  # queries a domain must be cited in
    max_domains: int = 5


@dataclass(frozen=True)
class Opportunity:
    target_brand: str
    provider: str
    intent: str
    gap_type: str
    metric: str
    target_value: float
    competitor_value: float | None  # strongest competitor listed, if any
    competitors: tuple[tuple[str, float], ...]
    compared_provider: str | None  # weaker_on_provider only
    compared_value: float | None
    queries: int
    answers: int
    supporting_query_ids: tuple[str, ...]
    supporting_domains: tuple[tuple[str, int], ...]  # (domain, queries citing it)
    observed: str
    hypothesis: str
    caveats: tuple[str, ...]


def detect_gaps(answers: Iterable[Answer], brand_ids: Sequence[str], target_brand: str,
                rules: GapRules = GapRules()) -> list[Opportunity]:
    if target_brand not in brand_ids:
        raise ValueError(f"target brand {target_brand!r} is not among the brands")
    answers = list(answers)
    return (_competitor_gaps(answers, brand_ids, target_brand, rules)
            + _provider_gaps(answers, brand_ids, target_brand, rules))


def cited_domains(answers: Iterable[Answer], brands: Iterable[str], field: str,
                  rules: GapRules = GapRules()) -> tuple[tuple[str, int], ...]:
    """Domains cited in answers where any of the brands is visible, counted once per query."""
    brands = set(brands)
    queries: dict[str, set[str]] = defaultdict(set)
    for a in answers:
        if brands & getattr(a, field):
            for d in a.domains:
                queries[d].add(a.query_id)
    ranked = sorted(((d, len(q)) for d, q in queries.items() if len(q) >= rules.min_domain_queries),
                    key=lambda x: (-x[1], x[0]))
    return tuple(ranked[:rules.max_domains])


def _groups(answers: list[Answer]) -> dict[tuple[str, str], list[Answer]]:
    groups: dict[tuple[str, str], list[Answer]] = defaultdict(list)
    for a in answers:
        groups[(a.provider, a.intent)].append(a)
    return groups


def _query_rates(answers: list[Answer], brand: str, field: str) -> dict[str, float]:
    runs: dict[str, list[bool]] = defaultdict(list)
    for a in answers:
        runs[a.query_id].append(brand in getattr(a, field))
    return {q: mean(r) for q, r in runs.items()}


def _pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def _caveats(intent: str, metric: str, providers: set[str]) -> tuple[str, ...]:
    out = []
    if intent == "brand_comparison":
        out.append("Brand comparison queries name specific brands, so rates partly reflect which brands "
                   "each query names.")
    if metric == "recommendation_rate":
        out.append("Recommended-or-not labels changed for up to 7% of mentions between classification runs.")
    if metric == "citation_rate" and "gemini" in providers:
        out.append("Gemini returns only a domain per citation, so citations on shared domains (zoho.com, "
                   "freshworks.com, monday.com) are not attributed and those brands are undercounted.")
    if metric == "citation_rate" and len(providers) > 1:
        out.append("Providers cite sources at different rates overall, not only for this brand.")
    return tuple(out)


def _hypothesis(domains: tuple[tuple[str, int], ...]) -> str:
    if not domains:
        return ("Untested hypothesis: competitors are favoured in these answers for reasons this data "
                "does not show; no outside domain recurs often enough to point to.")
    names = ", ".join(d for d, _ in domains)
    return (f"Untested hypothesis: sources such as {names} appear in answers where these competitors "
            "are visible and may inform them. Being cited together does not show that these sources "
            "caused the difference.")


def _competitor_gaps(answers: list[Answer], brand_ids: Sequence[str], target_brand: str,
                     rules: GapRules) -> list[Opportunity]:
    rates: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in brand_rates(answers, brand_ids, target_brand):
        rates[(r["provider"], r["intent"])][r["brand_id"]] = r
    groups = _groups(answers)

    out = []
    for (provider, intent), brands in sorted(rates.items()):
        target = brands[target_brand]
        if target["queries"] < rules.min_queries:
            continue
        group = groups[(provider, intent)]
        for gap_type, metric in COMPETITOR_GAPS.items():
            field = METRICS[metric][0]
            mine = _query_rates(group, target_brand, field)
            competitors, support = [], set()
            for brand, r in brands.items():
                if brand == target_brand or r[metric] - target[metric] < rules.min_gap:
                    continue
                theirs = _query_rates(group, brand, field)
                ahead = {q for q in mine if theirs[q] > mine[q]}
                if len(ahead) >= rules.min_queries_ahead:
                    competitors.append((brand, r[metric]))
                    support |= ahead
            if not competitors:
                continue
            competitors.sort(key=lambda c: (-c[1], c[0]))
            domains = cited_domains(group, [b for b, _ in competitors], field, rules)
            rivals = ", ".join(f"{b} {_pct(v)}" for b, v in competitors)
            out.append(Opportunity(
                target_brand=target_brand, provider=provider, intent=intent, gap_type=gap_type, metric=metric,
                target_value=target[metric], competitor_value=competitors[0][1], competitors=tuple(competitors),
                compared_provider=None, compared_value=None,
                queries=target["queries"], answers=target["answers"],
                supporting_query_ids=tuple(sorted(support)), supporting_domains=domains,
                observed=(f"In {provider} answers to {intent} queries ({target['queries']} queries, "
                          f"{target['answers']} answers), {target_brand} has a {LABELS[metric]} of "
                          f"{_pct(target[metric])}, against {rivals}."),
                hypothesis=_hypothesis(domains), caveats=_caveats(intent, metric, {provider})))
    return out


def _provider_gaps(answers: list[Answer], brand_ids: Sequence[str], target_brand: str,
                   rules: GapRules) -> list[Opportunity]:
    matched = matched_queries(answers)
    providers = sorted({a.provider for a in matched})
    if len(providers) != 2:
        return []
    rates = {(r["provider"], r["intent"], r["brand_id"]): r for r in brand_rates(matched, brand_ids, target_brand)}
    groups = _groups(matched)

    out = []
    for intent in sorted({a.intent for a in matched}):
        first, second = (rates[(p, intent, target_brand)] for p in providers)
        if min(first["queries"], second["queries"]) < rules.min_queries:
            continue
        for metric, (field, _) in METRICS.items():
            if abs(first[metric] - second[metric]) < rules.min_gap:
                continue
            weak, strong = providers if first[metric] < second[metric] else providers[::-1]
            weak_row, strong_row = rates[(weak, intent, target_brand)], rates[(strong, intent, target_brand)]
            weak_q = _query_rates(groups[(weak, intent)], target_brand, field)
            strong_q = _query_rates(groups[(strong, intent)], target_brand, field)
            support = sorted(q for q in weak_q if weak_q[q] < strong_q[q])
            if len(support) < rules.min_queries_ahead:
                continue
            competitors = sorted(((b, rates[(weak, intent, b)][metric]) for b in brand_ids
                                  if b != target_brand
                                  and rates[(weak, intent, b)][metric] - weak_row[metric] >= rules.min_gap),
                                 key=lambda c: (-c[1], c[0]))
            domains = cited_domains(groups[(weak, intent)], [b for b, _ in competitors], field, rules)
            ahead = f" On {weak}, " + ", ".join(f"{b} {_pct(v)}" for b, v in competitors) + "." if competitors else ""
            out.append(Opportunity(
                target_brand=target_brand, provider=weak, intent=intent, gap_type=PROVIDER_GAP, metric=metric,
                target_value=weak_row[metric], competitor_value=competitors[0][1] if competitors else None,
                competitors=tuple(competitors), compared_provider=strong, compared_value=strong_row[metric],
                queries=weak_row["queries"], answers=weak_row["answers"],
                supporting_query_ids=tuple(support), supporting_domains=domains,
                observed=(f"On the {weak_row['queries']} {intent} queries both providers answered, {target_brand} "
                          f"has a {LABELS[metric]} of {_pct(weak_row[metric])} on {weak} and "
                          f"{_pct(strong_row[metric])} on {strong}.{ahead}"),
                hypothesis=_hypothesis(domains) if competitors else (
                    f"Untested hypothesis: {weak} draws on different sources or answer styles for these "
                    f"queries; no competitor is ahead of {target_brand} there."),
                caveats=_caveats(intent, metric, set(providers))))
    return out
