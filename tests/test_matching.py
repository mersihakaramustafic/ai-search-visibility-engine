"""Deterministic brand matching and the matching config (no LLM, no database)."""

from __future__ import annotations

import copy

import pytest

from brand_fixtures import BRANDS
from visibility_extractor.config import ExtractionConfigError, load_config, parse_config
from visibility_extractor.matching import BrandMatcher, aggregate_mentions

RAW = {
    "matching": {
        "ambiguous_aliases": {"close": ["Close"]},
        "extra_ambiguous_forms": {"monday_crm": ["monday.com", "Monday.com", "monday", "Monday"],
                                  "freshsales": ["Freshworks"]},
        "exclusions": ["SAP Sales Cloud", "Oracle Sales Cloud", "Bigin by Zoho", "Zoho (?:One|Desk|Books|Bigin)"],
        "max_candidates_per_brand": 20,
        "snippet_chars": 30,
    }
}

CONFIG = parse_config(RAW).matching
MATCHER = BrandMatcher(BRANDS, CONFIG)


def found(text, masks=None):
    """(confirmed, candidates) as (brand_id, matched text) pairs, in answer order."""
    result = MATCHER.match(text, masks)
    return ([(o.brand_id, o.text) for o in result.confirmed],
            [(o.brand_id, o.text) for o in result.candidates])


# --- finding brands -------------------------------------------------------------


def test_multiple_brands_in_one_answer():
    confirmed, candidates = found("Top picks: **HubSpot**, Pipedrive, Attio and Zendesk Sell.")
    assert confirmed == [("hubspot", "HubSpot"), ("pipedrive", "Pipedrive"),
                         ("attio", "Attio"), ("zendesk", "Zendesk Sell")]
    assert candidates == []


def test_longest_alias_wins_when_matches_overlap():
    confirmed, _ = found("Salesforce Sales Cloud vs HubSpot Sales Hub vs Zoho CRM Plus")
    assert confirmed == [("salesforce", "Salesforce Sales Cloud"), ("hubspot", "HubSpot Sales Hub"),
                         ("zoho_crm", "Zoho CRM Plus")]


def test_aliases_ignore_case_but_need_whole_words():
    confirmed, _ = found("hubspot works, HUBSPOT too, and HubSpot's API. SuperHubSpot and Pipedrives do not.")
    assert confirmed == [("hubspot", "hubspot"), ("hubspot", "HUBSPOT"), ("hubspot", "HubSpot")]


def test_answer_without_tracked_brands():
    assert found("A CRM stores customer data and tracks deals.") == ([], [])


# --- masking ------------------------------------------------------------------------


def test_brand_names_inside_urls_do_not_count():
    confirmed, _ = found("See https://www.hubspot.com/pricing and https://pipedrive.com/en for prices.")
    assert confirmed == []


def test_exclusions_block_other_products():
    confirmed, _ = found("SAP Sales Cloud, Oracle Sales Cloud, Zoho One, Zoho Desk and Bigin by Zoho "
                         "are not Zoho CRM or Sales Cloud.")
    assert confirmed == [("zoho_crm", "Zoho CRM"), ("salesforce", "Sales Cloud")]


def test_bare_zoho_still_counts():
    confirmed, _ = found("Alternatives: Zoho, Close CRM.")
    assert confirmed == [("zoho_crm", "Zoho"), ("close", "Close CRM")]


def test_caller_supplied_spans_are_masked():
    # Stage 3 will pass citation markers like ([hubspot.com](url)) this way.
    text = "HubSpot is popular. ([hubspot.com](https://www.hubspot.com/x))"
    marker_start = text.index("([")
    confirmed, _ = found(text, [(marker_start, len(text))])
    assert confirmed == [("hubspot", "HubSpot")]


def test_without_the_mask_a_citation_label_would_count():
    confirmed, _ = found("HubSpot is popular. ([hubspot.com](https://www.hubspot.com/x))")
    assert confirmed == [("hubspot", "HubSpot"), ("hubspot", "hubspot")]


# --- ambiguous forms -------------------------------------------------------------


def test_close_is_a_candidate_never_a_confirmed_mention():
    confirmed, candidates = found("Close more deals. ### 3. Close\nNutshell, Close, Capsule.")
    assert confirmed == []
    assert candidates == [("close", "Close"), ("close", "Close"), ("close", "Close")]


def test_ambiguous_forms_are_case_sensitive():
    _, candidates = found("Deals close faster when you close the loop.")
    assert candidates == []


def test_unambiguous_forms_of_an_ambiguous_brand_still_confirm():
    confirmed, candidates = found("Try Close CRM, close.com or Close.io.")
    assert confirmed == [("close", "Close CRM"), ("close", "close.com"), ("close", "Close.io")]
    assert candidates == []


def test_extra_forms_are_candidates_but_full_aliases_confirm():
    confirmed, candidates = found("monday sales CRM is built on monday.com; review it every Monday. "
                                  "Freshworks CRM and Freshworks.")
    assert confirmed == [("monday_crm", "monday sales CRM"), ("freshsales", "Freshworks CRM")]
    assert candidates == [("monday_crm", "monday.com"), ("monday_crm", "Monday"), ("freshsales", "Freshworks")]


def test_candidates_over_the_cap_are_dropped_and_counted():
    config = parse_config({"matching": {**RAW["matching"], "max_candidates_per_brand": 2}}).matching
    result = BrandMatcher(BRANDS, config).match("Close. Close. Close. monday.")
    assert [o.text for o in result.candidates] == ["Close", "Close", "monday"]
    assert result.candidates_dropped == 1


# --- aggregation ---------------------------------------------------------------------


def test_aggregate_counts_ranks_and_offsets():
    text = "Pipedrive first. Then HubSpot. Pipedrive again, and HubSpot CRM."
    mentions = aggregate_mentions(text, MATCHER.match(text).confirmed, snippet_chars=10)
    assert [(m.brand_id, m.mention_count, m.first_mention_rank, m.first_char_offset, m.matched_alias)
            for m in mentions] == [("pipedrive", 2, 1, 0, "Pipedrive"), ("hubspot", 2, 2, 22, "HubSpot")]
    assert mentions[1].context_snippet == "rst. Then HubSpot. Pipedriv"  # 10 chars each side
    assert all(m.match_method == "alias" for m in mentions)


def test_brand_found_only_through_candidates_is_llm_disambiguated():
    text = "Pick Close. Close CRM is good."
    result = MATCHER.match(text)
    both = aggregate_mentions(text, result.confirmed + result.candidates, snippet_chars=10)
    only_candidate = aggregate_mentions("Pick Close.", MATCHER.match("Pick Close.").candidates, snippet_chars=10)
    assert both[0].match_method == "alias" and both[0].mention_count == 2
    assert only_candidate[0].match_method == "llm_disambiguated"


# --- config -------------------------------------------------------------------------------


def test_committed_config_is_valid_and_fits_the_brands():
    config = load_config()
    BrandMatcher(BRANDS, config.matching)  # raises if it names unknown brands or aliases
    assert config.matching.ambiguous_aliases == {"close": ("Close",)}


@pytest.mark.parametrize("change, message", [
    ({"exclusions": ["Zoho ("]}, "invalid regex"),
    ({"exclusions": "SAP"}, "list of regex strings"),
    ({"ambiguous_aliases": {"close": []}}, "non-empty lists"),
    ({"max_candidates_per_brand": 0}, "positive integer"),
    ({"snippet_chars": True}, "positive integer"),
])
def test_invalid_config_is_reported(change, message):
    raw = copy.deepcopy(RAW)
    raw["matching"].update(change)
    with pytest.raises(ExtractionConfigError) as exc:
        parse_config(raw)
    assert any(message in e for e in exc.value.errors)


def test_config_must_match_the_brands_table():
    bad = parse_config({"matching": {**RAW["matching"], "ambiguous_aliases": {"close": ["Clsoe"], "nope": ["X"]}}})
    with pytest.raises(ValueError, match="unknown brand_id 'nope'.*'Clsoe' is not one of its aliases"):
        BrandMatcher(BRANDS, bad.matching)
