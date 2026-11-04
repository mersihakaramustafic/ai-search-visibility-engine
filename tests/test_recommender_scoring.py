"""Priority scoring of validated recommendations (pure, no API or database)."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import yaml

from recommender_fixtures import pack
from visibility_recommender.config import DEFAULT_CONFIG_PATH, RecommendationConfigError, load_config, parse_config
from visibility_recommender.output import GenerationOutput, Recommendation
from visibility_recommender.scoring import band, gap_size, rank, score_output, score_recommendation

CONFIG = load_config().scoring
RAW = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
FULL_EVIDENCE = ["citation:10", "domain:gartner.com", "query:Q003", "metric:salesforce"]


def rec(category="directory_listing", evidence_ids=FULL_EVIDENCE, impact="medium", effort="medium",
        confidence="medium") -> Recommendation:
    return Recommendation(category=category, title="Update the Gartner CRM listing",
                          action="Review and update the listing on the cited Gartner CRM page.",
                          rationale="Gartner is cited in the supporting queries where Salesforce is ahead.",
                          evidence_ids=list(evidence_ids), expected_impact=impact, effort=effort,
                          confidence=confidence)


def test_shipped_config_has_scoring():
    assert CONFIG.version == "p1"
    assert CONFIG.weights == {"impact": 0.35, "evidence": 0.30, "relevance": 0.25, "effort": 0.10}


def test_worked_example():
    # gap 0.20 of 0.40 -> 0.5, reach 3/20, all four evidence kinds, 20 queries, directory fits cited_more fully.
    s = score_recommendation(rec(), 0, pack(), CONFIG)
    assert s.features == pytest.approx({"gap": 0.5, "reach": 0.15, "llm_impact": 0.5, "coverage": 1.0, "sample": 1.0,
                                        "llm_confidence": 0.5, "fit": 1.0, "intent": 1.0, "metric": 0.8,
                                        "effort": 0.5})
    assert s.parts == pytest.approx({"impact": 0.43, "evidence": 0.925, "relevance": 0.94, "effort": 0.5})
    assert (s.score, s.band, s.scoring_version, s.index) == (71.3, "high", "p1", 0)
    assert s.opportunity_id == pack().opportunity_id


def test_probe_example_from_the_proposal():
    # The probe's numbers: 25% vs 45% citation rate, one supporting query out of 20.
    probe = pack(target_value=0.25, competitor_value=0.45, competitors=(("salesforce", 0.45),),
                 supporting_query_ids=("Q003",))
    assert score_recommendation(rec(), 0, probe, CONFIG).score == pytest.approx(70.6)  # 14.35 + 27.75 + 23.5 + 5


def test_same_input_gives_same_score():
    assert score_recommendation(rec(), 0, pack(), CONFIG) == score_recommendation(rec(), 0, pack(), CONFIG)


def test_llm_ratings_move_the_score_by_at_most_21_5_points():
    best = score_recommendation(rec(impact="high", confidence="high", effort="low"), 0, pack(), CONFIG)
    worst = score_recommendation(rec(impact="low", confidence="low", effort="high"), 0, pack(), CONFIG)
    assert best.score - worst.score == pytest.approx(21.5)


def test_gap_is_capped_and_provider_gap_uses_the_other_provider():
    assert score_recommendation(rec(), 0, pack(competitor_value=0.95), CONFIG).features["gap"] == 1.0
    o = pack(gap_type="weaker_on_provider", competitor_value=None, competitors=(), compared_provider="gemini",
             compared_value=0.4333).opportunity
    assert gap_size(o) == pytest.approx(0.2)


def test_target_metric_adds_no_coverage():
    s = score_recommendation(rec(evidence_ids=["metric:hubspot", "query:Q003"]), 0, pack(), CONFIG)
    assert s.features["coverage"] == pytest.approx(0.25)


def test_offsite_action_without_a_source_has_its_fit_halved():
    no_source = ["query:Q003", "metric:salesforce"]
    assert score_recommendation(rec(evidence_ids=no_source), 0, pack(), CONFIG).features["fit"] == 0.5
    owned = score_recommendation(rec("owned_content_optimization", no_source), 0, pack(), CONFIG)
    assert owned.features["fit"] == 0.8  # owned content acts on the brand's own site, so no source is needed


def test_fit_follows_gap_type():
    s = score_recommendation(rec("community_visibility"), 0, pack(gap_type="target_mentioned_less"), CONFIG)
    assert s.features["fit"] == 1.0


def test_unknown_intent_is_a_clear_error():
    with pytest.raises(ValueError, match="no intent value for 'new_intent'"):
        score_recommendation(rec(), 0, pack(intent="new_intent"), CONFIG)


def test_bands():
    assert [band(x, CONFIG) for x in (70, 69.99, 50, 49.99)] == ["high", "medium", "medium", "low"]


def test_score_output_and_rank_with_tie_breaks():
    output = GenerationOutput(opportunity_id=pack().opportunity_id, insufficient_evidence=False,
                              recommendations=[rec(effort="high"), rec(effort="low"), rec(effort="high")])
    scores = score_output(output, pack(), CONFIG)
    assert [s.index for s in rank(scores)] == [1, 0, 2]  # equal scores keep the LLM's order
    bigger_gap = replace(scores[2], gap=0.3)
    assert rank([scores[0], bigger_gap])[0] is bigger_gap


def raw_with(change) -> dict:
    raw = copy.deepcopy(RAW)
    change(raw["scoring"])
    return raw


@pytest.mark.parametrize("change, message", [
    (lambda s: s["weights"].update(impact=0.5), "scoring.weights must add up to 1"),
    (lambda s: s["evidence"]["kinds"].pop("query"), "scoring.evidence.kinds must have exactly"),
    (lambda s: s["relevance"]["fit"].pop("weaker_on_provider"), "scoring.relevance.fit must have exactly"),
    (lambda s: s["relevance"]["fit"]["competitor_cited_more"].update(directory_listing=1.5), "values must be numbers"),
    (lambda s: s["impact"].update(full_gap=0), "full_gap"),
    (lambda s: s["bands"].update(medium=80), "scoring.bands"),
    (lambda s: s.update(version="has space"), "scoring.version"),
])
def test_invalid_scoring_config_is_rejected(change, message):
    with pytest.raises(RecommendationConfigError, match=message):
        parse_config(raw_with(change))


def test_scoring_section_is_optional():
    raw = copy.deepcopy(RAW)
    del raw["scoring"]
    assert parse_config(raw).scoring is None
