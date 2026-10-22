#!/usr/bin/env python3
"""Validation report for one extractor version, as Markdown, computed from the database (read-only)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.engine import Connection  # noqa: E402

from visibility_db.config import DEFAULT_ENV_VAR, DatabaseConfigError, make_engine  # noqa: E402
from visibility_extractor.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

RUN = """
    SELECT count(*) AS extractions, min(created_at) AS first_at, max(created_at) AS last_at,
           string_agg(DISTINCT strategy, ', ') AS strategies,
           string_agg(DISTINCT classification_model_requested, ', ') AS model_requested,
           string_agg(DISTINCT classification_model_id, ', ') AS model_reported,
           string_agg(DISTINCT prompt_version, ', ') AS prompt_versions,
           string_agg(DISTINCT schema_version, ', ') AS schema_versions
    FROM observation_extractions WHERE extractor_version = :v
"""

COVERAGE = """
    SELECT o.provider,
           count(*) AS answers,
           count(e.extraction_id) AS processed,
           count(*) FILTER (WHERE e.status = 'succeeded') AS succeeded,
           count(*) FILTER (WHERE e.status = 'failed') AS failed,
           count(*) FILTER (WHERE e.validation_status = 'valid_with_warnings') AS with_warnings,
           count(*) FILTER (WHERE e.validation_status = 'invalid') AS invalid,
           count(*) FILTER (WHERE e.classification_status = 'succeeded') AS classified,
           count(*) FILTER (WHERE e.classification_status = 'not_needed') AS no_call_needed
    FROM observations o
    LEFT JOIN observation_extractions e ON e.observation_id = o.observation_id AND e.extractor_version = :v
    WHERE o.status = 'succeeded'
    GROUP BY o.provider ORDER BY o.provider
"""

FAILURES = """
    SELECT e.observation_id, o.provider, e.error_type, e.classification_attempts AS attempts,
           left(e.error_message, 160) AS reason
    FROM observation_extractions e JOIN observations o USING (observation_id)
    WHERE e.extractor_version = :v AND e.status = 'failed' ORDER BY 1
"""

ISSUES = """
    SELECT issue, count(*) AS answers
    FROM observation_extractions e, jsonb_array_elements_text(e.validation_issues) AS issue
    WHERE e.extractor_version = :v AND e.status = 'succeeded'
    GROUP BY issue ORDER BY 2 DESC LIMIT 20
"""

TARGET = """
    SELECT o.provider,
           count(*) AS answers,
           count(*) FILTER (WHERE e.target_brand_mentioned) AS target_mentioned,
           count(*) FILTER (WHERE e.target_brand_recommended) AS target_recommended,
           round(avg(e.competitor_brands_mentioned), 2) AS avg_competitors
    FROM observation_extractions e JOIN observations o USING (observation_id)
    WHERE e.extractor_version = :v AND e.status = 'succeeded'
    GROUP BY o.provider ORDER BY o.provider
"""

BRANDS = """
    SELECT m.brand_id,
           count(*) FILTER (WHERE o.provider = 'openai') AS openai_answers,
           count(*) FILTER (WHERE o.provider = 'gemini') AS gemini_answers,
           sum(m.mention_count) AS occurrences,
           count(*) FILTER (WHERE m.match_method = 'llm_disambiguated') AS only_via_llm,
           count(*) FILTER (WHERE m.sentiment = 'positive') AS positive,
           count(*) FILTER (WHERE m.sentiment = 'neutral') AS neutral,
           count(*) FILTER (WHERE m.sentiment = 'negative') AS negative,
           count(*) FILTER (WHERE m.recommendation = 'primary') AS primary_rec,
           count(*) FILTER (WHERE m.recommendation = 'conditional') AS conditional_rec,
           count(*) FILTER (WHERE m.recommendation = 'none') AS not_recommended
    FROM brand_mentions m JOIN observations o USING (observation_id)
    WHERE m.extractor_version = :v
    GROUP BY m.brand_id ORDER BY count(*) DESC
"""

CANDIDATES = """
    SELECT count(*) FILTER (WHERE candidate_count > 0) AS answers_with_candidates,
           coalesce(sum(candidate_count), 0) AS candidates,
           coalesce(sum((SELECT count(*) FROM jsonb_array_elements(classification_output -> 'candidates') c
                         WHERE (c ->> 'is_brand')::boolean)) FILTER (WHERE classification_status = 'succeeded'), 0)
               AS accepted
    FROM observation_extractions WHERE extractor_version = :v AND status = 'succeeded'
"""

CITATIONS = """
    SELECT o.provider, c.citation_source,
           count(*) AS citations,
           count(c.domain) AS with_domain,
           count(c.brand_id) AS linked_to_brand,
           count(*) FILTER (WHERE c.is_target_brand_domain) AS target_domain
    FROM citations c JOIN observations o USING (observation_id)
    WHERE c.extractor_version = :v
    GROUP BY 1, 2 ORDER BY 1, 2
"""

NO_CITATIONS = """
    SELECT o.provider, count(*) AS answers_without_citations
    FROM observation_extractions e JOIN observations o USING (observation_id)
    WHERE e.extractor_version = :v AND e.status = 'succeeded'
      AND NOT EXISTS (SELECT 1 FROM citations c
                      WHERE c.observation_id = e.observation_id AND c.extractor_version = e.extractor_version)
    GROUP BY 1 ORDER BY 1
"""

USAGE = """
    SELECT count(*) FILTER (WHERE classification_attempts > 0) AS answers_sent,
           coalesce(sum(classification_attempts), 0) AS calls,
           coalesce(sum(classification_input_tokens), 0) AS input_tokens,
           coalesce(sum(classification_cached_input_tokens), 0) AS cached_input_tokens,
           coalesce(sum(classification_output_tokens), 0) AS output_tokens,
           coalesce(sum(classification_reasoning_tokens), 0) AS reasoning_tokens,
           coalesce(sum(classification_estimated_cost_usd), 0) AS cost_usd,
           avg(classification_estimated_cost_usd) FILTER (WHERE classification_attempts > 0) AS cost_per_answer,
           round(avg(classification_latency_ms)) AS avg_latency_ms,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY classification_latency_ms) AS p95_latency_ms
    FROM observation_extractions WHERE extractor_version = :v
"""

REMAINING = """
    SELECT
        (SELECT count(*) FROM observations o WHERE o.status = 'succeeded' AND NOT EXISTS (
            SELECT 1 FROM observation_extractions e
            WHERE e.observation_id = o.observation_id AND e.extractor_version = :v)) AS unprocessed_answers,
        (SELECT count(*) FROM observation_extractions WHERE extractor_version = :v AND status = 'failed')
            AS failed_answers,
        (SELECT count(*) FROM collection_requests WHERE status IN ('pending', 'failed')) AS uncollected_requests
"""

AGREEMENT = """
    SELECT count(*) AS shared_mentions,
           count(*) FILTER (WHERE a.sentiment = b.sentiment) AS same_sentiment,
           count(*) FILTER (WHERE a.recommendation = b.recommendation) AS same_recommendation,
           count(*) FILTER (WHERE (a.recommendation <> 'none') = (b.recommendation <> 'none')) AS same_recommended_or_not
    FROM brand_mentions a JOIN brand_mentions b USING (observation_id, brand_id)
    WHERE a.extractor_version = :v AND b.extractor_version = :other
      AND a.sentiment IS NOT NULL AND b.sentiment IS NOT NULL
"""


def rows(conn: Connection, sql: str, **params: Any) -> list[dict[str, Any]]:
    return [dict(r._mapping) for r in conn.execute(text(sql), params)]


def fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, float) or type(value).__name__ == "Decimal":
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(data: list[dict[str, Any]], empty: str = "_None._") -> str:
    if not data:
        return empty
    head = list(data[0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(fmt(r[h]) for h in head) + " |" for r in data]
    return "\n".join(lines)


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "–"


def build(conn: Connection, version: str, compare_with: str | None) -> str:
    run = rows(conn, RUN, v=version)[0]
    if not run["extractions"]:
        raise SystemExit(f"no extractions stored for version {version!r}")
    coverage = rows(conn, COVERAGE, v=version)
    usage = rows(conn, USAGE, v=version)[0]
    remaining = rows(conn, REMAINING, v=version)[0]
    candidates = rows(conn, CANDIDATES, v=version)[0]
    answers = sum(r["answers"] for r in coverage)
    succeeded = sum(r["succeeded"] for r in coverage)
    per_answer = float(usage["cost_per_answer"] or 0)

    out = [
        f"# Extraction validation report: `{version}`",
        "",
        f"Generated {fmt(datetime.now(timezone.utc))} by `scripts/extraction_report.py` from the database. "
        "Counts, tokens and latencies are measured. Costs are estimates from the prices in "
        "`configs/extraction.yaml`, which come from third-party listings.",
        "",
        "## Run",
        "",
        table([{"field": k, "value": v} for k, v in run.items()]),
        "",
        "## Coverage",
        "",
        f"{succeeded} of {answers} collected answers ({pct(succeeded, answers)}) have a usable extraction.",
        "",
        table(coverage),
        "",
        "## Failures",
        "",
        "Failed extractions store no mentions or citations; the rejected model output is kept in "
        "`observation_extractions.classification_output`. Rerun with `--retry-failed`.",
        "",
        table(rows(conn, FAILURES, v=version)),
        "",
        "## Validation warnings on stored extractions",
        "",
        table(rows(conn, ISSUES, v=version)),
        "",
        "## Target brand",
        "",
        table(rows(conn, TARGET, v=version)),
        "",
        "## Brand mentions",
        "",
        "One row in `brand_mentions` per brand per answer. `only_via_llm`: found only through ambiguous words "
        "the LLM confirmed.",
        "",
        table(rows(conn, BRANDS, v=version)),
        "",
        "## Ambiguous words",
        "",
        f"{fmt(candidates['candidates'])} ambiguous occurrences in {fmt(candidates['answers_with_candidates'])} "
        f"answers; the LLM accepted {fmt(candidates['accepted'])} as the brand "
        f"({pct(candidates['accepted'], candidates['candidates'])}).",
        "",
        "## Citations",
        "",
        "`provider_metadata`: citations the provider returned. `response_text`: URLs only written in the answer.",
        "",
        table(rows(conn, CITATIONS, v=version)),
        "",
        table(rows(conn, NO_CITATIONS, v=version)),
        "",
        "## Tokens and cost",
        "",
        "Cached input tokens are part of input tokens; reasoning tokens are part of output tokens.",
        "",
        table([{"metric": k, "value": v} for k, v in usage.items()]),
    ]

    if compare_with:
        agree = rows(conn, AGREEMENT, v=version, other=compare_with)[0]
        n = agree["shared_mentions"]
        out += [
            "",
            f"## Agreement with `{compare_with}`",
            "",
            f"Brand mentions labelled by both versions: {n}. Differences mix the change between the two "
            "versions with the model's run-to-run variation; with identical prompts they measure that variation.",
            "",
            table([
                {"label": "sentiment", "same": agree["same_sentiment"], "share": pct(agree["same_sentiment"], n)},
                {"label": "recommendation (primary / conditional / none)", "same": agree["same_recommendation"],
                 "share": pct(agree["same_recommendation"], n)},
                {"label": "recommended or not", "same": agree["same_recommended_or_not"],
                 "share": pct(agree["same_recommended_or_not"], n)},
            ]),
        ]

    todo = remaining["unprocessed_answers"] + remaining["failed_answers"]
    out += [
        "",
        "## Estimated cost of remaining work",
        "",
        f"Measured average: ${per_answer:.5f} per answer sent to the model.",
        "",
        table([
            {"work": "collected answers not yet extracted by this version, plus failed ones",
             "answers": todo, "estimated_cost_usd": round(todo * per_answer, 4)},
            {"work": "collection requests not yet collected (extraction cost only, once collected)",
             "answers": remaining["uncollected_requests"],
             "estimated_cost_usd": round(remaining["uncollected_requests"] * per_answer, 4)},
        ]),
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", help="Extractor version (default: the one in the config)")
    parser.add_argument("--compare-with", help="Another extractor version to measure label agreement against")
    parser.add_argument("--output", type=Path, help="Write the report here instead of printing it")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--db-env", default=DEFAULT_ENV_VAR)
    args = parser.parse_args(argv)
    try:
        version = args.version or load_config(args.config).extractor_version
        engine = make_engine(args.db_env)
    except DatabaseConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        report = build(conn, version, args.compare_with)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
