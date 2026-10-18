#!/usr/bin/env python3
"""Collect OpenAI and Gemini answers for pending collection requests.

Reads requests from collection_requests (joined to queries for the exact
text), calls each provider, and stores every API attempt in observations,
linked to a new collection_runs row. Settings live in configs/collection.yaml;
API keys and DATABASE_URL come from the environment or .env.

    pilot   only the configured 3-5 pilot queries (or --query-ids)
    full    every pending request

Usage:
    python scripts/collect_responses.py --mode pilot [--dry-run]
    python scripts/collect_responses.py --mode full [--provider openai] [--limit N]
                                        [--retry-failed] [--requeue-in-progress]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visibility_collector.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    SUPPORTED_PROVIDERS,
    CollectorConfigError,
    load_config,
    validate_pilot_query_ids,
)
from visibility_collector.logs import configure_logging, log  # noqa: E402
from visibility_collector.providers import build_client  # noqa: E402
from visibility_collector.repository import Repository  # noqa: E402
from visibility_collector.runner import Collector, Selection  # noqa: E402
from visibility_db.config import DEFAULT_ENV_VAR, DatabaseConfigError, describe_url, make_engine  # noqa: E402


def code_version() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--provider", action="append", choices=SUPPORTED_PROVIDERS,
                        help="Only this provider (repeatable). Default: every configured provider")
    parser.add_argument("--query-ids", help="Pilot only: comma-separated 3-5 query IDs overriding the config")
    parser.add_argument("--limit", type=int, help="Stop after this many requests per provider")
    parser.add_argument("--retry-failed", action="store_true", help="Also collect requests whose status is 'failed'")
    parser.add_argument("--requeue-in-progress", action="store_true",
                        help="First reset 'in_progress' requests (left by a crash) to 'pending'. "
                             "Only use when no other collector is running")
    parser.add_argument("--run-label", help="Label for the collection_runs row (default: <mode>-<UTC time>)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--db-env", default=DEFAULT_ENV_VAR, help="Environment variable holding the database URL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be collected; no API calls or writes")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    if args.query_ids and args.mode != "pilot":
        parser.error("--query-ids only applies to --mode pilot")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_config(args.config)
        providers = args.provider or list(config.providers)
        missing = [p for p in providers if p not in config.providers]
        if missing:
            raise CollectorConfigError([f"provider {p} is not configured in {args.config}" for p in missing])

        query_ids = None
        if args.mode == "pilot":
            query_ids = [q.strip() for q in args.query_ids.split(",")] if args.query_ids else list(config.pilot_query_ids)
            errors = validate_pilot_query_ids(query_ids)
            if errors:
                raise CollectorConfigError(errors)

        engine = make_engine(args.db_env)
        repo = Repository(engine)
        if query_ids:
            unknown = sorted(set(query_ids) - repo.existing_query_ids(query_ids))
            if unknown:
                raise CollectorConfigError([f"query IDs not in the queries table: {unknown}"])

        statuses = ["pending", "failed"] if args.retry_failed else ["pending"]
        selection = Selection(statuses=statuses, query_ids=query_ids, limit_per_provider=args.limit)
        print(f"Database: {describe_url(engine.url)} (from ${args.db_env})", file=sys.stderr)

        if args.dry_run:
            return dry_run(repo, providers, config, selection)

        # Build clients first: a missing API key fails before any row is written.
        clients = {p: build_client(config.providers[p]) for p in providers}
    except (CollectorConfigError, DatabaseConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for p in providers:
        if args.requeue_in_progress:
            n = repo.requeue_in_progress(p, query_ids)
            log(logging.INFO, "requeued in_progress requests", provider=p, count=n)

    run_label = args.run_label or f"{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    collector_config = {
        "mode": args.mode,
        "providers": {p: {"model": config.providers[p].model,
                          "max_output_tokens": config.providers[p].max_output_tokens,
                          "timeout_seconds": config.providers[p].timeout_seconds,
                          "concurrency": config.providers[p].concurrency,
                          **config.providers[p].options} for p in providers},
        "query_ids": query_ids,
        "statuses": statuses,
        "limit_per_provider": args.limit,
        "retry": vars(config.retry),
    }
    run_id = repo.create_run(run_label, collector_config, code_version())
    log(logging.INFO, "collection run started", run_id=run_id, run_label=run_label, mode=args.mode,
        providers=providers, query_ids=query_ids)

    collector = Collector(repo, clients, config.retry, {p: config.providers[p].concurrency for p in providers})
    try:
        summaries = collector.run(run_id, selection)
    except KeyboardInterrupt:
        repo.finish_run(run_id, "cancelled", "Interrupted by user")
        log(logging.WARNING, "collection run cancelled", run_id=run_id)
        return 130
    except Exception as e:
        repo.finish_run(run_id, "failed", f"{type(e).__name__}: {str(e)[:500]}")
        log(logging.ERROR, "collection run failed", run_id=run_id, error=type(e).__name__)
        raise

    notes = "; ".join(f"{p}: " + ", ".join(f"{k}={v}" for k, v in s.as_dict().items()) for p, s in summaries.items())
    stopped = any(s.stopped_reason for s in summaries.values())
    repo.finish_run(run_id, "failed" if stopped else "completed", notes)
    log(logging.INFO, "collection run finished", run_id=run_id, run_label=run_label,
        summary={p: s.as_dict() for p, s in summaries.items()})
    print(f"\nRun {run_id} ({run_label}):", file=sys.stderr)
    for p, s in summaries.items():
        print(f"  {p:7} " + ", ".join(f"{k}={v}" for k, v in s.as_dict().items()), file=sys.stderr)
    failed = any(s.failed or s.stopped_reason for s in summaries.values())
    return 2 if failed else 0


def dry_run(repo: Repository, providers: list[str], config, selection: Selection) -> int:
    for p in providers:
        tasks = repo.preview(p, selection.statuses, selection.query_ids, selection.limit_per_provider)
        counts = repo.status_counts(p, selection.query_ids)
        print(f"\n{p} (model {config.providers[p].model}): {len(tasks)} request(s) would be collected; "
              f"current statuses {counts}")
        for t in tasks[:20]:
            print(f"  {t.request_id:28} search={str(t.search_enabled).lower():5} {t.query_text[:70]}")
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more")
    print("\nDry run: no API calls made, nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
