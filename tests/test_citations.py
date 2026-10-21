"""Citation parsing from stored OpenAI and Gemini metadata (no LLM, no database)."""

from __future__ import annotations

import pytest

from brand_fixtures import BRANDS
from visibility_extractor.citations import DomainAttributor, extract_citations, normalize_domain
from visibility_extractor.config import ExtractionConfigError, load_config, parse_config
from visibility_extractor.matching import BrandMatcher

CONFIG = load_config()
ATTRIBUTOR = DomainAttributor(BRANDS, CONFIG.citations)
REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQ"


def marker(domain, url):
    return f"([{domain}]({url}))"


def openai_answer():
    """Text plus metadata shaped like a stored OpenAI observation."""
    hub = "https://www.hubspot.com/products/crm?utm_source=openai"
    zoho = "https://www.zoho.com/crm/zohocrm-pricing.html?utm_source=openai"
    text = ("For most small teams, **HubSpot** is the default pick. " + marker("hubspot.com", hub)
            + "\n| **Zoho CRM** | Cheapest paid tiers. " + marker("zoho.com", zoho) + " |\n"
            + "HubSpot again: " + marker("hubspot.com", hub))
    annotations = []
    position = 0
    for domain, url in (("hubspot.com", hub), ("zoho.com", zoho), ("hubspot.com", hub)):
        start = text.index(marker(domain, url), position)
        end = start + len(marker(domain, url))
        position = end
        annotations.append({"type": "url_citation", "url": url, "title": f"{domain} page",
                            "start_index": start, "end_index": end})
    return text, {"url_citations": annotations, "web_search_calls": [{"action": {"sources": [{"url": "x"}]}}]}


def gemini_answer():
    """Text plus metadata shaped like a stored Gemini observation."""
    text = ("### 1. Pipedrive\nLoved for its visual pipeline.\n### 2. Zoho CRM\nCheap.\n"
            "Pricing details: https://www.pipedrive.com/en/pricing.")
    meta = {"grounding_metadata": {
        "web_search_queries": ["best crm"],
        "grounding_chunks": [{"web": {"uri": REDIRECT + "1", "title": "pipedrive.com"}},
                             {"web": {"uri": REDIRECT + "2", "title": "zoho.com"}},
                             {"web": {"uri": REDIRECT + "3", "title": "knowledge.hubspot.com"}},
                             {"web": {"uri": REDIRECT + "4", "title": "Best CRMs of 2026"}}],
        "grounding_supports": [{"segment": {"text": "Loved"}, "grounding_chunk_indices": [0, 2]},
                               {"segment": {"text": "Cheap."}, "grounding_chunk_indices": [0]}],
        "search_entry_point": {"rendered_content": "<div/>"},
    }}
    return text, meta


def provider(result):
    return [c for c in result.citations if c.citation_source == "provider_metadata"]


def prose(result):
    return [c for c in result.citations if c.citation_source == "response_text"]


# --- OpenAI ---------------------------------------------------------------------------


def test_openai_citations_come_from_annotations():
    text, meta = openai_answer()
    result = extract_citations("openai", text, meta, ATTRIBUTOR)
    assert [(c.citation_position, c.domain, c.brand_id, c.is_target_brand_domain) for c in result.citations] == [
        (1, "hubspot.com", "hubspot", True), (2, "zoho.com", "zoho_crm", False), (3, "hubspot.com", "hubspot", True)]
    first = result.citations[0]
    assert first.provider_format == "openai_url_citation" and first.title == "hubspot.com page"
    assert first.url.endswith("utm_source=openai")  # stored unchanged
    assert result.warnings == []


def test_openai_search_sources_are_not_citations():
    text, meta = openai_answer()
    assert all(c.url != "x" for c in extract_citations("openai", text, meta, ATTRIBUTOR).citations)


def test_openai_marker_spans_are_returned_and_keep_markers_out_of_brand_matching():
    text, meta = openai_answer()
    result = extract_citations("openai", text, meta, ATTRIBUTOR)
    assert all(text[s:e].startswith("([") and text[s:e].endswith("))") for s, e in result.masked_spans)

    matcher = BrandMatcher(BRANDS, CONFIG.matching)
    with_masks = [o.text for o in matcher.match(text, result.masked_spans).confirmed]
    without = [o.text for o in matcher.match(text).confirmed]
    assert with_masks == ["HubSpot", "Zoho CRM", "HubSpot"]
    assert len(without) == 6  # hubspot.com / zoho.com labels inside the markers counted too


# --- Gemini -----------------------------------------------------------------------------


def test_gemini_citations_use_title_domain_and_never_guess():
    text, meta = gemini_answer()
    result = extract_citations("gemini", text, meta, ATTRIBUTOR)
    assert [(c.domain, c.brand_id, c.is_target_brand_domain) for c in provider(result)] == [
        ("pipedrive.com", "pipedrive", False),
        ("zoho.com", None, False),               # shared domain, and Gemini gives no path
        ("knowledge.hubspot.com", "hubspot", True),
        (None, None, None),                      # title is not a domain
    ]
    assert provider(result)[1].metadata["brand_attribution"].startswith("skipped")


def test_gemini_metadata_keeps_chunk_details():
    text, meta = gemini_answer()
    first = provider(extract_citations("gemini", text, meta, ATTRIBUTOR))[0]
    assert first.provider_format == "gemini_grounding_chunk"
    assert first.url == REDIRECT + "1"
    assert first.metadata == {"chunk_index": 0, "uri_is_redirect": True, "supporting_segments": 2}


# --- URLs written in the answer ------------------------------------------------------------


def test_prose_urls_are_kept_apart_from_provider_citations():
    text, meta = gemini_answer()
    urls = prose(extract_citations("gemini", text, meta, ATTRIBUTOR))
    assert [(u.url, u.domain, u.brand_id, u.provider_format, u.citation_position) for u in urls] == [
        ("https://www.pipedrive.com/en/pricing", "pipedrive.com", "pipedrive", "text_url", 5)]  # final "." dropped


def test_openai_marker_urls_are_not_repeated_as_prose():
    text, meta = openai_answer()
    assert prose(extract_citations("openai", text, meta, ATTRIBUTOR)) == []


def test_same_prose_url_twice_is_recorded_once():
    text = "See https://close.com/pricing and again https://close.com/pricing."
    urls = prose(extract_citations("gemini", text, None, ATTRIBUTOR))
    assert [(u.url, u.brand_id, u.metadata["char_offset"]) for u in urls] == [("https://close.com/pricing", "close", 4)]


# --- brand attribution on shared domains --------------------------------------------------


@pytest.mark.parametrize("url, brand_id", [
    ("https://www.zoho.com/crm/features.html", "zoho_crm"),
    ("https://help.zoho.com/portal/en/kb/crm/getting-started", "zoho_crm"),
    ("https://www.zoho.com/one/pricing.html", None),
    ("https://monday.com/crm/pricing", "monday_crm"),
    ("https://monday.com/pricing", None),
    ("https://www.freshworks.com/crm/pricing/", "freshsales"),
    ("https://crmsupport.freshworks.com/support/solutions", "freshsales"),
    ("https://www.freshworks.com/products/what-is-freshsales", "freshsales"),
    ("https://www.freshworks.com/freshdesk/", None),
    ("https://help.salesforce.com/s/articleView?id=1", "salesforce"),
    ("https://www.g2.com/products/hubspot-crm/reviews", None),
])
def test_brand_attribution(url, brand_id):
    [citation] = prose(extract_citations("gemini", f"Source: {url}", None, ATTRIBUTOR))
    assert citation.brand_id == brand_id


# --- missing and malformed metadata --------------------------------------------------------


@pytest.mark.parametrize("provider_name", ["openai", "gemini"])
def test_answer_without_citations(provider_name):
    result = extract_citations(provider_name, "Plain answer without links.", None, ATTRIBUTOR)
    assert result.citations == [] and result.warnings == [] and result.masked_spans == []


@pytest.mark.parametrize("provider_name, meta, warning", [
    ("openai", {"url_citations": "oops"}, "not a list"),
    ("openai", {"url_citations": [{"title": "no url"}, "junk"]}, "without url"),
    ("gemini", {"grounding_metadata": []}, "not an object"),
    ("gemini", {"grounding_metadata": {"grounding_chunks": [{"maps": {}}]}}, "no web uri"),
    ("openai", ["not", "a", "dict"], "not an object"),
    ("mistral", {"citations": [1]}, "no citation parser"),
])
def test_malformed_metadata_is_reported_not_guessed(provider_name, meta, warning):
    result = extract_citations(provider_name, "text", meta, ATTRIBUTOR)
    assert provider(result) == []
    assert any(warning in w for w in result.warnings)


@pytest.mark.parametrize("value, expected", [
    ("WWW.HubSpot.com", "hubspot.com"), ("blog.hubspot.com.", "blog.hubspot.com"),
    ("Best CRMs of 2026", None), ("", None), (None, None)])
def test_normalize_domain(value, expected):
    assert normalize_domain(value) == expected


# --- config ---------------------------------------------------------------------------------

MINIMAL = {"extractor_version": "test-v1", "matching": {"max_candidates_per_brand": 1, "snippet_chars": 1}}


def test_rules_must_name_known_brands():
    config = parse_config({**MINIMAL, "citations": {"brand_url_rules": {"zoho": ["/crm"]}}})
    with pytest.raises(ValueError, match="zoho"):
        DomainAttributor(BRANDS, config.citations)


def test_invalid_rules_are_reported():
    with pytest.raises(ExtractionConfigError, match="citations.brand_url_rules"):
        parse_config({**MINIMAL, "citations": {"brand_url_rules": {"zoho_crm": "/crm"}}})
