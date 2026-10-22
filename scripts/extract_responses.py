#!/usr/bin/env python3
"""Extract brand mentions and citations from stored answers: rules first, then the LLM where configured."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visibility_collector.logs import configure_logging, log  # noqa: E402
from visibility_collector.providers.base import ProviderError  # noqa: E402
from visibility_db.config import DEFAULT_ENV_VAR, DatabaseConfigError, describe_url, make_engine  # noqa: E402
from visibility_extractor.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    STRATEGIES,
    ExtractionConfigError,
    load_config,
)
from visibility_extractor.llm import LLMClassifier  # noqa: E402
from visibility_extractor.pipeline import Extraction, RulesExtractor  # noqa: E402
from visibility_extractor.repository import ExtractionRepository  # noqa: E402
from visibility_extractor.runner import ExtractionRunner  # noqa: E402

PROVIDERS = ["openai", "gemini"]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pilot", action="store_true",
                        help="Only the fixed pilot sample (pilot.observations_per_provider per provider)")
    parser.add_argument("--provider", action="append", choices=PROVIDERS,
                        help="Only this source provider (repeatable). Default: all")
    parser.add_argument("--observation-ids", help="Comma-separated observation IDs")
    parser.add_argument("--limit", type=int, help="Process at most this many observations")
    parser.add_argument("--strategy", choices=STRATEGIES, help="Override the configured strategy for every provider")
    parser.add_argument("--retry-failed", action="store_true", help="Also redo extractions of this version that failed")
    parser.add_argument("--reprocess", action="store_true",
                        help="Redo everything selected, even if this version already processed it (rows are replaced)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--db-env", default=DEFAULT_ENV_VAR, help="Environment variable holding the database URL")
    parser.add_argument("--dry-run", action="store_true", help="Run the rules and show what the LLM would get; "
                                                               "no API calls, nothing written")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.pilot and args.observation_ids:
        parser.error("--pilot and --observation-ids cannot be combined")
    if args.observation_ids:
        try:
            args.observation_ids = [int(i) for i in args.observation_ids.split(",")]
        except ValueError:
            parser.error("--observation-ids must be comma-separated integers")
    return args


def describe(e: Extraction) -> str:
    if e.status == "failed":
        attempts = f" after {e.llm.attempts} LLM attempt(s)" if e.llm else ""
        return f"FAILED{attempts} {e.error_type}: {e.error_message[:200]}"
    llm = e.llm.status if e.llm else "-"
    brands = ", ".join(f"{m.brand_id}x{m.mention_count}" + (f"[{m.sentiment[:3]}/{m.recommendation}]"
                                                             if m.sentiment else "") for m in e.mentions) or "-"
    sources = Counter(c.citation_source for c in e.citations)
    rec = "-" if e.target_brand_recommended is None else str(e.target_brand_recommended).lower()
    line = (f"{e.validation_status:19} llm={llm:10} target={str(e.target_brand_mentioned).lower():5} "
            f"rec={rec:5} cand={e.candidate_count:<3} "
            f"cit={sources['provider_metadata']}+{sources['response_text']}  {brands}")
    return line + "".join(f"\n      ! {i}" for i in e.validation_issues)


def log_extraction(e: Extraction, provider: str) -> None:
    r = e.llm
    log(logging.INFO, "extracted", observation_id=e.observation_id, provider=provider, strategy=e.strategy,
        status=e.status, validation_status=e.validation_status, llm_status=r and r.status,
        model=r and r.model_id, attempts=r and r.attempts, input_tokens=r and r.input_tokens,
        cached_input_tokens=r and r.cached_input_tokens, output_tokens=r and r.output_tokens,
        latency_ms=r and r.latency_ms, cost_usd=r and r.estimated_cost_usd, error_type=e.error_type)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_config(args.config)
        if args.strategy:
            config = replace(config, strategy={p: args.strategy for p in PROVIDERS})
        engine = make_engine(args.db_env)
        repo = ExtractionRepository(engine)
        rules = RulesExtractor(repo.load_brands(), config)
        uses_llm = "rules_and_llm" in config.strategy.values()
        classifier = None
        if uses_llm and not args.dry_run:
            if config.llm is None:
                raise ExtractionConfigError(["strategy rules_and_llm needs an llm section"])
            classifier = LLMClassifier.from_config(config.llm)
            reported = classifier.check_access()  # fail before any work, never fall back to another model
            print(f"LLM: {config.llm.model} (API reports {reported})", file=sys.stderr)
    except (ExtractionConfigError, DatabaseConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ProviderError as e:
        print(f"error: model {config.llm.model} is not usable: {e.error_type}: {e.message}", file=sys.stderr)
        return 1

    print(f"Database: {describe_url(engine.url)} (from ${args.db_env})", file=sys.stderr)
    providers = args.provider or PROVIDERS
    ids = repo.pilot_ids(providers, config.pilot_per_provider) if args.pilot else args.observation_ids
    observations = repo.select_observations(config.extractor_version, providers, ids, args.limit,
                                            args.reprocess, args.retry_failed)
    print(f"Extractor {config.extractor_version}, strategy {config.strategy}: "
          f"{len(observations)} observation(s) to process")

    runner = ExtractionRunner(rules, config, classifier)
    if args.dry_run:
        return dry_run(runner, observations)

    totals: Counter[str] = Counter()
    cost = 0.0
    workers = config.llm.concurrency if classifier else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(runner.process, obs): obs for obs in observations}
        try:
            for future in as_completed(futures):
                obs = futures[future]
                extraction = future.result()
                repo.save(extraction)
                log_extraction(extraction, obs.provider)
                totals[extraction.validation_status or "failed"] += 1
                if extraction.llm:
                    totals[f"llm_{extraction.llm.status}"] += 1
                    totals["llm_calls"] += extraction.llm.attempts
                    cost += extraction.llm.estimated_cost_usd or 0
                print(f"  obs {obs.observation_id:<5} {obs.provider:6} {describe(extraction)}")
        except ProviderError as e:  # fatal: same failure for every remaining observation
            pool.shutdown(cancel_futures=True)
            print(f"\nStopped: {e.error_type}: {e.message}. Everything finished before this is saved.",
                  file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            pool.shutdown(cancel_futures=True)
            print("\nInterrupted. Everything finished before this is saved; rerun to continue.", file=sys.stderr)
            return 130

    print("\nThis run: " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())) + f", cost=${cost:.4f}")
    print("Stored for this version: " + ", ".join(
        f"{k}={v}" for k, v in repo.extraction_summary(config.extractor_version).items()))
    return 2 if totals["failed"] or totals["invalid"] else 0


def dry_run(runner: ExtractionRunner, observations) -> int:
    calls = 0
    for obs in observations:
        e = replace(runner.rules.extract(obs), strategy=runner.config.strategy_for(obs.provider))
        needs = runner.needs_llm(obs, e)
        calls += needs
        print(f"  obs {obs.observation_id:<5} {obs.provider:6} {e.strategy:13} llm_call={str(needs).lower():5} "
              f"{describe(e)}")
    print(f"\nDry run: {calls} LLM call(s) would be made (more if retried). No API calls, nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
