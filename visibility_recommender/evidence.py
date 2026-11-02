"""The facts an LLM may use for one opportunity, each with an id it must cite."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from visibility_analysis.gaps import LABELS, Opportunity

MAX_CITATIONS_PER_DOMAIN = 2


@dataclass(frozen=True)
class QueryInfo:
    query_id: str
    text: str
    topic: str
    contains_brand_name: bool


@dataclass(frozen=True)
class CitationRef:
    citation_id: int
    query_id: str
    provider: str
    domain: str
    url: str
    title: str | None


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str  # 'metric' | 'query' | 'domain' | 'citation'
    fact: str
    data: dict[str, Any]


@dataclass(frozen=True)
class EvidencePack:
    opportunity_id: str
    opportunity: Opportunity
    items: tuple[EvidenceItem, ...]
    insufficient_reason: str | None  # set when the LLM should not be asked

    @property
    def ids(self) -> set[str]:
        return {i.evidence_id for i in self.items}

    def by_id(self, evidence_id: str) -> EvidenceItem:
        return next(i for i in self.items if i.evidence_id == evidence_id)

    def facts(self) -> list[dict[str, str]]:
        return [{"id": i.evidence_id, "fact": i.fact} for i in self.items]


def opportunity_id(o: Opportunity, extractor_version: str) -> str:
    return f"{o.target_brand}:{o.provider}:{o.intent}:{o.gap_type}:{o.metric}@{extractor_version}"


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def insufficient_reason(o: Opportunity) -> str | None:
    if not o.competitors and not o.supporting_domains:
        return ("No competitor is ahead of the target brand and no outside domain recurs in enough queries, "
                "so there is nothing specific to act on.")
    return None


def build_pack(o: Opportunity, extractor_version: str, queries: Mapping[str, QueryInfo],
               citations: Iterable[CitationRef]) -> EvidencePack:
    label, group = LABELS[o.metric], f"{o.provider} answers to {o.intent} queries"
    items = [EvidenceItem(f"metric:{o.target_brand}", "metric",
                          f"{o.target_brand} has a {label} of {pct(o.target_value)} in {group} "
                          f"({o.queries} queries, {o.answers} answers).",
                          {"brand_id": o.target_brand, "provider": o.provider, "value": o.target_value})]
    if o.compared_provider is not None:
        items.append(EvidenceItem(f"metric:{o.target_brand}@{o.compared_provider}", "metric",
                                  f"{o.target_brand} has a {label} of {pct(o.compared_value)} in "
                                  f"{o.compared_provider} answers to the same {o.intent} queries.",
                                  {"brand_id": o.target_brand, "provider": o.compared_provider,
                                   "value": o.compared_value}))
    for brand, value in o.competitors:
        items.append(EvidenceItem(f"metric:{brand}", "metric", f"{brand} has a {label} of {pct(value)} in {group}.",
                                  {"brand_id": brand, "provider": o.provider, "value": value}))
    for q in o.supporting_query_ids:
        info = queries[q]
        items.append(EvidenceItem(f"query:{q}", "query",
                                  f"Query {q} ({info.topic}) shows the gap: \"{info.text}\"",
                                  {"query_id": q, "topic": info.topic, "contains_brand_name": info.contains_brand_name}))
    support = set(o.supporting_query_ids)
    by_domain: dict[str, list[CitationRef]] = defaultdict(list)
    for c in sorted(citations, key=lambda c: c.citation_id):
        if c.provider == o.provider and c.query_id in support and c.url not in {x.url for x in by_domain[c.domain]}:
            by_domain[c.domain].append(c)
    for domain, n in o.supporting_domains:
        items.append(EvidenceItem(f"domain:{domain}", "domain",
                                  f"{domain} is cited in {n} {o.provider} {o.intent} queries where the listed "
                                  "competitors are visible.", {"domain": domain, "queries": n}))
        for c in by_domain[domain][:MAX_CITATIONS_PER_DOMAIN]:
            items.append(EvidenceItem(f"citation:{c.citation_id}", "citation",
                                      f"{o.provider} cited {c.url}" + (f" (\"{c.title}\")" if c.title else "")
                                      + f" when answering {c.query_id}.",
                                      {"citation_id": c.citation_id, "url": c.url, "domain": domain,
                                       "query_id": c.query_id}))
    return EvidencePack(opportunity_id(o, extractor_version), o, tuple(items), insufficient_reason(o))
