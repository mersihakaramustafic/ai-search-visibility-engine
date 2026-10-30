"""Visibility gap rules on synthetic answers."""

from __future__ import annotations

from itertools import count

import pytest

from visibility_analysis.gaps import GapRules, cited_domains, detect_gaps
from visibility_analysis.metrics import Answer

BRANDS = ["hubspot", "pipedrive", "salesforce"]
_ids = count(1)


def make(n=10, provider="openai", intent="use_case", runs=1, first=1,
         mentioned=None, recommended=None, cited=None, domains=None):
    """Queries Q<first>..; each spec maps brand -> query numbers where it is visible."""
    out = []
    for i in range(first, first + n):
        def pick(spec):
            return frozenset(b for b, qs in (spec or {}).items() if i in qs)
        for _ in range(runs):
            out.append(Answer(next(_ids), f"Q{i:03d}", provider, intent, pick(mentioned), pick(recommended),
                              pick(cited), frozenset((domains or {}).get(i, ()))))
    return out


def gaps(answers, gap_type=None, target="hubspot"):
    return [g for g in detect_gaps(answers, BRANDS, target) if gap_type in (None, g.gap_type)]


def test_competitor_recommended_more():
    answers = make(recommended={"hubspot": {1, 2, 3}, "pipedrive": range(1, 7)},
                   domains={4: {"g2.com"}, 5: {"g2.com"}, 6: {"g2.com", "reddit.com"}, 7: {"g2.com"}})
    [g] = gaps(answers)
    assert (g.provider, g.intent, g.gap_type, g.metric) == ("openai", "use_case", "competitor_recommended_more",
                                                            "recommendation_rate")
    assert (g.target_value, g.competitor_value, g.competitors) == (0.3, 0.6, (("pipedrive", 0.6),))
    assert (g.queries, g.answers, g.supporting_query_ids) == (10, 10, ("Q004", "Q005", "Q006"))
    assert g.supporting_domains == (("g2.com", 3),)  # Q007 has no visible competitor; reddit only once
    assert "30%" in g.observed and "pipedrive 60%" in g.observed
    assert g.hypothesis.startswith("Untested hypothesis") and "does not show" in g.hypothesis


def test_small_difference_is_ignored():
    answers = make(n=40, recommended={"hubspot": range(1, 11), "pipedrive": range(11, 24)})  # 25% vs 32.5%
    assert gaps(answers) == []


def test_gap_from_too_few_queries_is_ignored():
    assert gaps(make(recommended={"pipedrive": {1, 2}})) == []


@pytest.mark.parametrize("n, found", [(9, 0), (10, 1)])
def test_minimum_queries_per_group(n, found):
    assert len(gaps(make(n=n, recommended={"pipedrive": range(1, n + 1)}))) == found


def test_mention_and_citation_gaps_list_every_competitor_ahead():
    answers = make(mentioned={"hubspot": {1}, "pipedrive": range(1, 9), "salesforce": range(1, 6)},
                   cited={"salesforce": range(1, 5)})
    by_type = {g.gap_type: g for g in gaps(answers)}
    assert set(by_type) == {"target_mentioned_less", "competitor_cited_more"}
    assert by_type["target_mentioned_less"].competitors == (("pipedrive", 0.8), ("salesforce", 0.5))
    assert by_type["target_mentioned_less"].supporting_query_ids == tuple(f"Q00{i}" for i in range(2, 9))
    assert by_type["competitor_cited_more"].competitors == (("salesforce", 0.4),)


def test_repeated_runs_count_once():
    answers = make(n=12, runs=3, recommended={"pipedrive": range(1, 13)}, domains={i: {"g2.com"} for i in (1, 2, 3)})
    [g] = gaps(answers)
    assert (g.queries, g.answers, g.supporting_domains) == (12, 36, (("g2.com", 3),))


def test_cited_domains_threshold_and_order():
    answers = make(recommended={"pipedrive": range(1, 11)},
                   domains={1: {"b.com", "a.com"}, 2: {"b.com", "a.com"}, 3: {"b.com", "a.com"}, 4: {"b.com"},
                            5: {"c.com"}, 6: {"c.com"}})
    assert cited_domains(answers, ["pipedrive"], "recommended") == (("b.com", 4), ("a.com", 3))
    assert cited_domains(answers, ["salesforce"], "recommended") == ()
    assert cited_domains(answers, ["pipedrive"], "recommended", GapRules(max_domains=1)) == (("b.com", 4),)


def test_weaker_on_provider_uses_matched_queries_only():
    answers = (make(provider="openai", cited={"hubspot": range(1, 11)})
               + make(provider="gemini", cited={"hubspot": {1}, "pipedrive": range(1, 6)})
               + make(n=5, first=11, provider="gemini", cited={"hubspot": range(11, 16)}))  # only gemini answered
    [g] = gaps(answers, "weaker_on_provider")
    assert (g.provider, g.compared_provider, g.metric) == ("gemini", "openai", "citation_rate")
    assert (g.target_value, g.compared_value, g.queries) == (0.1, 1.0, 10)
    assert g.competitors == (("pipedrive", 0.5),)
    assert g.supporting_query_ids == tuple(f"Q{i:03d}" for i in range(2, 11))
    assert any("different rates overall" in c for c in g.caveats)


def test_no_provider_gap_without_two_providers():
    assert gaps(make(cited={"hubspot": range(1, 11)}), "weaker_on_provider") == []


def test_brand_comparison_caveat():
    [g] = gaps(make(intent="brand_comparison", recommended={"pipedrive": range(1, 11)}))
    assert any("name specific brands" in c for c in g.caveats)


def test_unknown_target_brand_is_rejected():
    with pytest.raises(ValueError, match="not among the brands"):
        detect_gaps(make(), BRANDS, "attio")
