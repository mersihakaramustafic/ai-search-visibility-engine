"""Evidence packs built from opportunities (no LLM, no database)."""

from __future__ import annotations

from recommender_fixtures import pack


def test_opportunity_id_is_stable_and_readable():
    assert pack().opportunity_id == "hubspot:openai:category_discovery:competitor_cited_more:citation_rate@rules-llm-v3"
    assert pack().opportunity_id == pack().opportunity_id
    assert pack(provider="gemini").opportunity_id != pack().opportunity_id


def test_items_cover_metrics_queries_domains_and_citations():
    p = pack()
    assert [i.evidence_id for i in p.items] == [
        "metric:hubspot", "metric:salesforce", "query:Q003", "query:Q004", "query:Q005",
        "domain:gartner.com", "citation:10", "citation:12", "domain:g2.com", "citation:20"]
    assert p.by_id("metric:hubspot").fact == (
        "hubspot has a citation rate of 23% in openai answers to category_discovery queries (20 queries, 60 answers).")
    assert p.by_id("citation:20").data["url"] == "https://www.g2.com/categories/crm"
    assert p.insufficient_reason is None


def test_citations_are_limited_deduplicated_and_from_supporting_answers():
    urls = [i.data["url"] for i in pack().items if i.kind == "citation"]
    assert len(urls) == len(set(urls))  # citation 11 repeats citation 10's URL
    assert not any("gemini-only" in u or "unsupported-query" in u or "third" in u for u in urls)


def test_provider_gap_includes_the_other_provider():
    p = pack(gap_type="weaker_on_provider", compared_provider="gemini", compared_value=0.8667)
    assert p.by_id("metric:hubspot@gemini").data["value"] == 0.8667


def test_insufficient_evidence_without_competitors_or_domains():
    p = pack(gap_type="weaker_on_provider", competitors=(), competitor_value=None, supporting_domains=(),
             compared_provider="gemini", compared_value=0.8)
    assert "nothing specific to act on" in p.insufficient_reason
    assert pack(supporting_domains=()).insufficient_reason is None  # a competitor ahead is still specific
