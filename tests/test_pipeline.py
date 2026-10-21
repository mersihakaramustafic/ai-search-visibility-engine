"""Rules-only extraction of whole observations (no LLM, no database)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from brand_fixtures import BRANDS
from test_citations import gemini_answer, openai_answer
from visibility_extractor.config import ExtractionConfigError, load_config, parse_config
from visibility_extractor.matching import MentionSummary
from visibility_extractor.pipeline import Observation, RulesExtractor, validate

CONFIG = load_config()
EXTRACTOR = RulesExtractor(BRANDS, CONFIG)
BRAND_IDS = {b.brand_id for b in BRANDS}


def run(provider, text, meta=None, observation_id=1):
    return EXTRACTOR.extract(Observation(observation_id, provider, text, meta))


def test_openai_observation():
    text, meta = openai_answer()
    e = run("openai", text, meta)
    assert (e.status, e.validation_status, e.validation_issues) == ("succeeded", "valid", [])
    assert e.extractor_version == CONFIG.extractor_version
    assert [(m.brand_id, m.mention_count, m.first_mention_rank) for m in e.mentions] == [
        ("hubspot", 2, 1), ("zoho_crm", 1, 2)]  # citation markers not counted
    assert len(e.citations) == 3
    assert (e.target_brand_mentioned, e.competitor_brands_mentioned, e.candidate_count) == (True, 1, 0)


def test_gemini_observation_without_target_brand():
    text, meta = gemini_answer()
    e = run("gemini", text, meta)
    assert [m.brand_id for m in e.mentions] == ["pipedrive", "zoho_crm"]
    assert (e.target_brand_mentioned, e.competitor_brands_mentioned) == (False, 2)
    assert [c.citation_source for c in e.citations] == ["provider_metadata"] * 4 + ["response_text"]


def test_ambiguous_forms_are_counted_not_stored():
    e = run("gemini", "Close is great. Try monday.com too.")
    assert e.mentions == [] and e.candidate_count == 2
    assert e.target_brand_mentioned is False


def test_answer_with_nothing_still_produces_an_extraction():
    e = run("openai", "A CRM stores customer data.")
    assert (e.status, e.validation_status, e.mentions, e.citations) == ("succeeded", "valid", [], [])
    assert (e.target_brand_mentioned, e.competitor_brands_mentioned, e.candidate_count) == (False, 0, 0)


def test_metadata_warnings_make_it_valid_with_warnings():
    e = run("openai", "HubSpot.", {"url_citations": "oops"})
    assert e.validation_status == "valid_with_warnings"
    assert any("not a list" in i for i in e.validation_issues)
    assert [m.brand_id for m in e.mentions] == ["hubspot"]


def test_dropped_candidates_are_a_warning():
    config = parse_config({**_raw(), "matching": {**_raw()["matching"], "max_candidates_per_brand": 1}})
    e = RulesExtractor(BRANDS, config).extract(Observation(1, "openai", "Close. Close. Close.", None))
    assert e.candidate_count == 1
    assert e.validation_issues == ["2 ambiguous occurrence(s) over the per-brand cap dropped"]


def test_unexpected_error_becomes_a_failed_extraction():
    e = run("openai", None)  # response_text is NOT NULL in the DB, so this can only be a bug
    assert (e.status, e.validation_status, e.error_type) == ("failed", None, "TypeError")
    assert e.mentions == [] and e.citations == []


# --- validation ------------------------------------------------------------------------------


def mention(**changes):
    base = MentionSummary("hubspot", 1, 4, 1, "HubSpot", "Try HubSpot", "alias")
    return replace(base, **changes)


def test_valid_mention_passes():
    assert validate("Try HubSpot", [mention()], [], BRAND_IDS) == []


@pytest.mark.parametrize("changes, error", [
    ({"first_char_offset": 0}, "text at offset 0 is 'Try Hub'"),
    ({"brand_id": "nope"}, "unknown brand 'nope'"),
    ({"first_mention_rank": 2}, "ranks are not 1..n"),
    ({"mention_count": 0}, "mention_count 0"),
])
def test_validation_catches_inconsistent_mentions(changes, error):
    errors = validate("Try HubSpot", [mention(**changes)], [], BRAND_IDS)
    assert any(error in e for e in errors)


def test_invalid_result_stores_nothing(monkeypatch):
    monkeypatch.setattr("visibility_extractor.pipeline.validate", lambda *a: ["broken"])
    text, meta = openai_answer()
    e = run("openai", text, meta)
    assert (e.status, e.validation_status, e.validation_issues) == ("succeeded", "invalid", ["broken"])
    assert e.mentions == [] and e.citations == [] and e.target_brand_mentioned is None


# --- config ------------------------------------------------------------------------------------


def _raw():
    return {"extractor_version": "test-v1", "matching": {"ambiguous_aliases": {"close": ["Close"]},
                                                    "max_candidates_per_brand": 20, "snippet_chars": 10}}


@pytest.mark.parametrize("version", [None, "", "rules v1", 3])
def test_extractor_version_is_required(version):
    with pytest.raises(ExtractionConfigError, match="extractor_version"):
        parse_config({**_raw(), "extractor_version": version})
