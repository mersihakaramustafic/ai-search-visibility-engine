"""Brand visibility rates from per-answer facts."""

from __future__ import annotations

from itertools import count

import pytest

from visibility_analysis.metrics import Answer, brand_rates, matched_queries

_ids = count(1)


def answer(query_id="Q001", provider="openai", intent="use_case", mentioned=(), recommended=(), cited=()):
    return Answer(next(_ids), query_id, provider, intent,
                  frozenset(mentioned), frozenset(recommended), frozenset(cited))


def one(rows, **match):
    [row] = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return row


def test_rates_and_counts():
    answers = [
        answer("Q001", mentioned={"hubspot", "pipedrive"}, recommended={"hubspot"}, cited={"hubspot"}),
        answer("Q002", mentioned={"hubspot"}),
        answer("Q003", mentioned={"pipedrive"}, recommended={"pipedrive"}),
        answer("Q004"),
    ]
    rows = brand_rates(answers, ["hubspot", "pipedrive"], "hubspot", by=())
    hubspot = one(rows, brand_id="hubspot")
    assert (hubspot["queries"], hubspot["answers"], hubspot["is_target_brand"]) == (4, 4, True)
    assert (hubspot["mentioned_answers"], hubspot["recommended_answers"], hubspot["cited_answers"]) == (2, 1, 1)
    assert (hubspot["mention_rate"], hubspot["recommendation_rate"], hubspot["citation_rate"]) == (0.5, 0.25, 0.25)
    pipedrive = one(rows, brand_id="pipedrive")
    assert (pipedrive["is_target_brand"], pipedrive["mention_rate"], pipedrive["citation_rate"]) == (False, 0.5, 0)


def test_repeated_runs_are_averaged_per_query():
    # Q001: mentioned in 2 of 3 runs; Q002: one run, not mentioned.
    answers = [answer("Q001", mentioned={"hubspot"}), answer("Q001", mentioned={"hubspot"}),
               answer("Q001"), answer("Q002")]
    row = one(brand_rates(answers, ["hubspot"], "hubspot", by=()), brand_id="hubspot")
    assert (row["queries"], row["answers"], row["mentioned_answers"]) == (2, 4, 2)
    assert row["mention_rate"] == pytest.approx((2 / 3 + 0) / 2)  # pooled would be 2/4


def test_grouped_by_provider_and_intent():
    answers = [answer("Q001", "openai", "use_case", mentioned={"hubspot"}),
               answer("Q001", "gemini", "use_case"),
               answer("Q002", "gemini", "pricing_purchase", mentioned={"hubspot"})]
    rows = brand_rates(answers, ["hubspot"], "hubspot")
    assert [(r["provider"], r["intent"], r["answers"], r["mention_rate"]) for r in rows] == [
        ("gemini", "pricing_purchase", 1, 1), ("gemini", "use_case", 1, 0), ("openai", "use_case", 1, 1)]


def test_brand_never_seen_gets_zero_rows():
    rows = brand_rates([answer()], ["attio"], "hubspot", by=("provider",))
    assert one(rows, brand_id="attio")["mention_rate"] == 0


def test_same_observation_twice_is_rejected():
    a = answer()
    with pytest.raises(ValueError, match="appears twice"):
        brand_rates([a, a], ["hubspot"], "hubspot")


def test_unknown_group_field_is_rejected():
    with pytest.raises(ValueError, match="cannot group by"):
        brand_rates([answer()], ["hubspot"], "hubspot", by=("run_number",))


def test_matched_queries_keeps_queries_every_provider_answered():
    both = [answer("Q001", "openai"), answer("Q001", "gemini"), answer("Q001", "gemini")]
    gemini_only = [answer("Q002", "gemini")]
    assert matched_queries(both + gemini_only) == both
