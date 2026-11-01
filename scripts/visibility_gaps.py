#!/usr/bin/env python3
"""Visibility gaps of a target brand against its competitors, as JSON and Markdown, from the database (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text  # noqa: E402

from visibility_analysis.gaps import LABELS, GapRules, Opportunity, detect_gaps  # noqa: E402
from visibility_analysis.metrics import Answer, matched_queries  # noqa: E402
from visibility_analysis.repository import AnalysisRepository  # noqa: E402
from visibility_db.config import DEFAULT_ENV_VAR, DatabaseConfigError, make_engine  # noqa: E402
from visibility_extractor.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

GAP_TITLES = {
    "competitor_recommended_more": "Competitors recommended more often",
    "competitor_cited_more": "Competitors' own sites cited more often",
    "target_mentioned_less": "Target brand mentioned less often",
    "weaker_on_provider": "Target brand weaker on one provider",
}


def samples(answers: list[Answer], rules: GapRules) -> list[dict[str, Any]]:
    """Queries and answers per provider and intent, and whether each group is large enough."""
    queries: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    matched: dict[tuple[str, str], set[str]] = defaultdict(set)
    for a in answers:
        queries[(a.provider, a.intent)].add(a.query_id)
        counts[(a.provider, a.intent)] += 1
    for a in matched_queries(answers):
        matched[(a.provider, a.intent)].add(a.query_id)
    return [{"provider": p, "intent": i, "queries": len(q), "answers": counts[(p, i)],
             "queries_both_providers": len(matched[(p, i)]),
             "competitor_gaps_checked": len(q) >= rules.min_queries,
             "provider_gap_checked": len(matched[(p, i)]) >= rules.min_queries}
            for (p, i), q in sorted(queries.items())]


def opportunity_dict(o: Opportunity) -> dict[str, Any]:
    d = asdict(o)
    d["competitors"] = [{"brand_id": b, "rate": round(v, 4)} for b, v in o.competitors]
    d["supporting_domains"] = [{"domain": dom, "queries": n} for dom, n in o.supporting_domains]
    for key in ("target_value", "competitor_value", "compared_value"):
        d[key] = None if d[key] is None else round(d[key], 4)
    d["supporting_query_ids"] = list(o.supporting_query_ids)
    d["caveats"] = list(o.caveats)
    return d


def build_json(answers: list[Answer], brand_ids: list[str], target: str, version: str,
               rules: GapRules, generated_at: datetime) -> dict[str, Any]:
    opportunities = detect_gaps(answers, brand_ids, target, rules)
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "extractor_version": version,
        "target_brand": target,
        "competitors": [b for b in brand_ids if b != target],
        "rules": asdict(rules),
        "answers": len(answers),
        "samples": samples(answers, rules),
        "opportunities": [opportunity_dict(o) for o in opportunities],
    }


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.0f}%"


def _md_table(head: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *("| " + " | ".join(str(c) for c in r) + " |" for r in rows)]


def build_markdown(report: dict[str, Any]) -> str:
    r, target, ops = report["rules"], report["target_brand"], report["opportunities"]
    out = [
        f"# Visibility gaps: `{target}` (`{report['extractor_version']}`)",
        "",
        f"Generated {report['generated_at']} by `scripts/visibility_gaps.py` from {report['answers']} answers. "
        "See [docs/visibility_gaps.md](../docs/visibility_gaps.md) for how gaps are found.",
        "",
        "**Observed** lines are measured from the stored answers. **Hypothesis** lines are untested readings; "
        "a domain cited alongside competitors is not evidence that it caused their visibility. "
        "No recommendations are made here.",
        "",
        "## Rules",
        "",
        *_md_table(["rule", "value"], [
            ["minimum queries per provider and intent", r["min_queries"]],
            ["minimum difference in rate", f"{100 * r['min_gap']:.0f} points"],
            ["minimum queries showing the difference", r["min_queries_ahead"]],
            ["minimum queries citing a supporting domain", r["min_domain_queries"]],
            ["supporting domains listed at most", r["max_domains"]],
        ]),
        "",
        "Each query counts once: its repeated runs are averaged before queries are averaged.",
        "",
        "## Sample sizes",
        "",
        *_md_table(["provider", "intent", "queries", "answers", "queries both providers answered",
                    "competitor gaps checked", "provider gap checked"],
                   [[s["provider"], s["intent"], s["queries"], s["answers"], s["queries_both_providers"],
                     "yes" if s["competitor_gaps_checked"] else "no, too few queries",
                     "yes" if s["provider_gap_checked"] else "no, too few queries"] for s in report["samples"]]),
        "",
        "## Summary",
        "",
    ]
    if not ops:
        out += ["_No gap met the rules._", ""]
        return "\n".join(out)
    out += _md_table(
        ["#", "gap", "provider", "intent", "metric", target, "competitor / other provider", "queries"],
        [[n, o["gap_type"], o["provider"], o["intent"], LABELS[o["metric"]], _pct(o["target_value"]),
          (f"{o['compared_provider']} {_pct(o['compared_value'])}" if o["compared_provider"]
           else f"{o['competitors'][0]['brand_id']} {_pct(o['competitor_value'])}"),
          o["queries"]] for n, o in enumerate(ops, 1)])
    for n, o in enumerate(ops, 1):
        competitors = ", ".join(f"{c['brand_id']} ({_pct(c['rate'])})" for c in o["competitors"]) or "none ahead"
        domains = ", ".join(f"{d['domain']} ({d['queries']} queries)" for d in o["supporting_domains"]) or (
            "none cited in enough queries" if o["competitors"] else "– (no competitor ahead)")
        out += [
            "",
            f"## {n}. {GAP_TITLES[o['gap_type']]}: {o['provider']}, {o['intent']}",
            "",
            f"- **Observed:** {o['observed']}",
            f"- **Competitors ahead:** {competitors}",
            f"- **Supporting queries ({len(o['supporting_query_ids'])}):** {', '.join(o['supporting_query_ids'])}",
            f"- **Domains cited where these competitors are visible:** {domains}",
            f"- **Hypothesis:** {o['hypothesis']}",
            *(f"- **Caveat:** {c}" for c in o["caveats"]),
        ]
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", help="Target brand id (default: the primary brand)")
    parser.add_argument("--version", help="Extractor version (default: the one in the config)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--db-env", default=DEFAULT_ENV_VAR)
    args = parser.parse_args(argv)
    try:
        version = args.version or load_config(args.config).extractor_version
        engine = make_engine(args.db_env)
    except DatabaseConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    @contextmanager
    def read_only():
        with engine.begin() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            yield conn

    repo = AnalysisRepository(engine, read_only)
    brand_ids, primary = repo.load_brands()
    target = args.target or primary
    if target not in brand_ids:
        print(f"error: unknown brand {target!r}; choose from {', '.join(brand_ids)}", file=sys.stderr)
        return 1
    answers = repo.load_answers(version)
    if not answers:
        print(f"error: no successful extractions stored for version {version!r}", file=sys.stderr)
        return 1

    report = build_json(answers, brand_ids, target, version, GapRules(), datetime.now(timezone.utc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"visibility_gaps_{target}_{version}"
    Path(f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path(f"{stem}.md").write_text(build_markdown(report), encoding="utf-8")
    print(f"{len(report['opportunities'])} opportunities for {target} from {len(answers)} answers; "
          f"wrote {stem}.json and .md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
