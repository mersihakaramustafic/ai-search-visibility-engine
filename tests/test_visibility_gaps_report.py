"""JSON and Markdown built by scripts/visibility_gaps.py from synthetic answers (no database)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from test_analysis_gaps import BRANDS, make
from visibility_analysis.gaps import GapRules
from visibility_gaps import build_json, build_markdown

NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def report(answers):
    return build_json(answers, BRANDS, "hubspot", "test-v1", GapRules(), NOW)


def test_json_fields_and_samples():
    answers = (make(recommended={"hubspot": {1, 2, 3}, "pipedrive": range(1, 7)}, domains={i: {"g2.com"} for i in (4, 5, 6)})
               + make(n=2, provider="openai", intent="pricing_purchase"))
    r = report(answers)
    json.dumps(r)  # serialisable
    assert (r["target_brand"], r["competitors"], r["answers"]) == ("hubspot", ["pipedrive", "salesforce"], 12)
    assert [(s["intent"], s["queries"], s["competitor_gaps_checked"]) for s in r["samples"]] == [
        ("pricing_purchase", 2, False), ("use_case", 10, True)]
    [o] = r["opportunities"]
    assert o["competitors"] == [{"brand_id": "pipedrive", "rate": 0.6}]
    assert o["supporting_domains"] == [{"domain": "g2.com", "queries": 3}]
    assert (o["target_value"], o["supporting_query_ids"]) == (0.3, ["Q004", "Q005", "Q006"])


def test_markdown_keeps_facts_and_hypotheses_apart():
    md = build_markdown(report(make(recommended={"pipedrive": range(1, 11)})))
    assert "## 1. Competitors recommended more often: openai, use_case" in md
    observed = next(line for line in md.splitlines() if line.startswith("- **Observed:**"))
    hypothesis = next(line for line in md.splitlines() if line.startswith("- **Hypothesis:**"))
    assert "Untested" not in observed and "Untested hypothesis" in hypothesis
    assert "no, too few queries" not in md


def test_markdown_without_gaps():
    md = build_markdown(report(make(n=3, recommended={"pipedrive": range(1, 4)})))
    assert "_No gap met the rules._" in md and "no, too few queries" in md
