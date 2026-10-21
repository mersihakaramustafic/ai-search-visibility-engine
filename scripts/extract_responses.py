#!/usr/bin/env python3
"""Extract brand mentions and citations from stored answers (rules only, no API calls)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visibility_db.config import DEFAULT_ENV_VAR, DatabaseConfigError, describe_url, make_engine  # noqa: E402
from visibility_extractor.config import DEFAULT_CONFIG_PATH, ExtractionConfigError, load_config  # noqa: E402
from visibility_extractor.pipeline import Extraction, RulesExtractor  # noqa: E402
from visibility_extractor.repository import ExtractionRepository  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", action="append", choices=["openai", "gemini"],
                        help="Only this source provider (repeatable). Default: all")
    parser.add_argument("--observation-ids", help="Comma-separated observation IDs")
    parser.add_argument("--limit", type=int, help="Process at most this many observations")
    parser.add_argument("--reprocess", action="store_true",
                        help="Also redo observations this extractor_version already processed (rows are replaced)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--db-env", default=DEFAULT_ENV_VAR, help="Environment variable holding the database URL")
    parser.add_argument("--dry-run", action="store_true", help="Extract and show results; write nothing")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.observation_ids:
        try:
            args.observation_ids = [int(i) for i in args.observation_ids.split(",")]
        except ValueError:
            parser.error("--observation-ids must be comma-separated integers")
    return args


def describe(e: Extraction) -> str:
    if e.status == "failed":
        return f"FAILED {e.error_type}: {e.error_message}"
    brands = ", ".join(f"{m.brand_id}x{m.mention_count}" for m in e.mentions) or "-"
    sources = Counter(c.citation_source for c in e.citations)
    line = (f"{e.validation_status:19} target={str(e.target_brand_mentioned).lower():5} "
            f"candidates={e.candidate_count:<3} citations={sources['provider_metadata']}+"
            f"{sources['response_text']}text  brands: {brands}")
    return line + "".join(f"\n      ! {i}" for i in e.validation_issues)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        engine = make_engine(args.db_env)
        repo = ExtractionRepository(engine)
        brands = repo.load_brands()
        extractor = RulesExtractor(brands, config)
    except (ExtractionConfigError, DatabaseConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Database: {describe_url(engine.url)} (from ${args.db_env})", file=sys.stderr)
    observations = repo.select_observations(config.extractor_version, args.provider, args.observation_ids,
                                            args.limit, args.reprocess)
    print(f"Extractor {config.extractor_version}: {len(observations)} observation(s) to process")

    totals: Counter[str] = Counter()
    for obs in observations:
        extraction = extractor.extract(obs)
        if not args.dry_run:
            repo.save(extraction)
        totals[extraction.validation_status or "failed"] += 1
        totals["brand_mentions"] += len(extraction.mentions)
        totals["citations"] += len(extraction.citations)
        if args.dry_run or extraction.validation_status != "valid":
            print(f"  obs {obs.observation_id:<5} {obs.provider:6} {describe(extraction)}")

    print("\nTotals: " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    if args.dry_run:
        print("Dry run: nothing written.")
    else:
        print("Stored for this version: " + ", ".join(
            f"{k}={v}" for k, v in repo.extraction_summary(config.extractor_version).items()))
    return 2 if totals["failed"] or totals["invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
