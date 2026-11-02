"""Validation of LLM recommendation output against the evidence pack (no LLM calls)."""

from __future__ import annotations

import json

import pytest

from recommender_fixtures import BRAND_DOMAINS, pack
from visibility_recommender.output import validate_output

PACK = pack()


def rec(**changes):
    r = {
        "category": "directory_listing",
        "title": "Strengthen HubSpot's Gartner reviews profile",
        "action": "Review and complete HubSpot's profile on the Gartner CRM reviews page cited for Q003 and Q005, "
                  "and add current customer reviews.",
        "rationale": "OpenAI cites gartner.com in answers where Salesforce is cited 43% of the time against "
                     "HubSpot's 23%; this may help, though citation does not prove cause.",
        "evidence_ids": ["metric:hubspot", "metric:salesforce", "domain:gartner.com", "citation:10"],
        "expected_impact": "medium", "effort": "low", "confidence": "medium",
    }
    return {**r, **changes}


def output(*recs, **changes):
    return json.dumps({"opportunity_id": PACK.opportunity_id, "insufficient_evidence": False,
                       "recommendations": list(recs) or [rec()], **changes})


def check(text):
    return validate_output(text, PACK, BRAND_DOMAINS)


def test_valid_output():
    parsed, errors = check(output())
    assert errors == [] and parsed.recommendations[0].category == "directory_listing"


def test_insufficient_evidence_reported_by_the_model():
    parsed, errors = check(output(recommendations=[], insufficient_evidence=True, insufficient_reason="Only metrics."))
    assert errors == [] and parsed.insufficient_evidence and parsed.recommendations == []


@pytest.mark.parametrize("text, expected", [
    ("not json", "Invalid JSON"),
    (output(rec(category="paid_ads")), "recommendations.0.category"),
    (output(rec(extra="x")), "Extra inputs"),
    (output(rec(effort="tiny")), "recommendations.0.effort"),
    (output(recommendations=[]), "at least one recommendation"),
    (output(recommendations=[], insufficient_evidence=True), "needs a reason"),
    (output(*[rec()] * 4), "at most 3"),
])
def test_schema_errors(text, expected):
    parsed, errors = check(text)
    assert parsed is None and any(expected in e for e in errors), errors


@pytest.mark.parametrize("changes, expected", [
    ({"evidence_ids": ["metric:hubspot", "domain:capterra.com"]}, "unknown evidence ids ['domain:capterra.com']"),
    ({"evidence_ids": ["metric:hubspot", "metric:salesforce"]}, "cites only metrics"),
    ({"rationale": "Salesforce is cited in 61% of answers, far more than HubSpot."}, "percentage '61%'"),
    ({"action": "Get listed on capterra.com, which ranks CRMs for buyers in this category."}, "domain 'capterra.com'"),
    ({"action": "Update the page at https://www.gartner.com/made-up-page for HubSpot's CRM."}, "URL"),
    ({"rationale": "This guarantees that HubSpot appears in more OpenAI answers about CRM."}, "certain outcome"),
    ({"rationale": "Doing this will increase HubSpot's citation rate in OpenAI answers."}, "certain outcome"),
])
def test_fabricated_or_overclaiming_content_is_rejected(changes, expected):
    parsed, errors = check(output(rec(**changes)))
    assert parsed is None and any(expected in e for e in errors), errors


def test_evidence_that_is_in_the_pack_is_accepted():
    text = output(rec(action="Ask Gartner to refresh the page at https://www.gartner.com/reviews/market/crm and "
                             "check g2.com and hubspot.com listings for the 20% citation difference.",
                      evidence_ids=["query:Q003", "citation:10", "domain:g2.com"]))
    assert check(text)[1] == []


def test_wrong_opportunity_id():
    parsed, errors = check(output(opportunity_id="hubspot:gemini:use_case:x:y@v"))
    assert parsed is None and "does not match" in errors[0]
