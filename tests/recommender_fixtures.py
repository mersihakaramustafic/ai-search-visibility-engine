"""A competitor gap and a provider gap with their queries and citations, shared by recommender tests."""

from __future__ import annotations

from visibility_analysis.gaps import Opportunity
from visibility_recommender.evidence import CitationRef, QueryInfo, build_pack

VERSION = "rules-llm-v3"
BRAND_DOMAINS = {"hubspot.com", "salesforce.com", "pipedrive.com"}


def opportunity(**changes) -> Opportunity:
    fields = dict(
        target_brand="hubspot", provider="openai", intent="category_discovery", gap_type="competitor_cited_more",
        metric="citation_rate", target_value=0.2333, competitor_value=0.4333,
        competitors=(("salesforce", 0.4333),), compared_provider=None, compared_value=None,
        queries=20, answers=60, supporting_query_ids=("Q003", "Q004", "Q005"),
        supporting_domains=(("gartner.com", 3), ("g2.com", 3)),
        observed="In openai answers ...", hypothesis="Untested hypothesis: ...", caveats=())
    return Opportunity(**{**fields, **changes})


QUERIES = {q: QueryInfo(q, f"Which CRM is best? ({q})", "market_overview", False) for q in ("Q003", "Q004", "Q005")}

CITATIONS = [
    CitationRef(10, "Q003", "openai", "gartner.com", "https://www.gartner.com/reviews/market/crm", "CRM Reviews"),
    CitationRef(11, "Q004", "openai", "gartner.com", "https://www.gartner.com/reviews/market/crm", "CRM Reviews"),
    CitationRef(12, "Q005", "openai", "gartner.com", "https://www.gartner.com/en/sales/crm", None),
    CitationRef(13, "Q004", "openai", "gartner.com", "https://www.gartner.com/third", None),
    CitationRef(20, "Q003", "openai", "g2.com", "https://www.g2.com/categories/crm", "Best CRM Software"),
    CitationRef(30, "Q003", "gemini", "g2.com", "https://www.g2.com/gemini-only", None),  # other provider
    CitationRef(40, "Q099", "openai", "g2.com", "https://www.g2.com/unsupported-query", None),  # not a supporting query
]


def pack(**changes):
    return build_pack(opportunity(**changes), VERSION, QUERIES, CITATIONS)
