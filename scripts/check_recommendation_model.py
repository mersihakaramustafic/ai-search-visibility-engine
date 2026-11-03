#!/usr/bin/env python3
"""Check that the configured recommendation model is usable. Never switches to another model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visibility_analysis.gaps import Opportunity  # noqa: E402
from visibility_collector.providers.base import ProviderError  # noqa: E402
from visibility_extractor.llm import estimate_cost  # noqa: E402
from visibility_recommender.config import DEFAULT_CONFIG_PATH, RecommendationConfigError, load_config  # noqa: E402
from visibility_recommender.evidence import CitationRef, QueryInfo, build_pack  # noqa: E402
from visibility_recommender.llm import make_generator  # noqa: E402
from visibility_recommender.output import validate_output  # noqa: E402

# A small synthetic opportunity: checks access, prompt and schema, not the quality of real recommendations.
PROBE = build_pack(
    Opportunity(
        target_brand="hubspot", provider="openai", intent="category_discovery", gap_type="competitor_cited_more",
        metric="citation_rate", target_value=0.25, competitor_value=0.45, competitors=(("salesforce", 0.45),),
        compared_provider=None, compared_value=None, queries=20, answers=60, supporting_query_ids=("Q900",),
        supporting_domains=(("g2.com", 3),), observed="Probe: synthetic numbers.",
        hypothesis="Untested hypothesis: probe.", caveats=()),
    "probe",
    {"Q900": QueryInfo("Q900", "What are the most popular CRM platforms right now?", "market_overview", False)},
    [CitationRef(1, "Q900", "openai", "g2.com", "https://www.g2.com/categories/crm", "Best CRM Software")],
)
BRAND_DOMAINS = {"hubspot.com", "salesforce.com"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--probe", action="store_true", help="Also make one small paid generation call")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        generator = make_generator(config.llm)
        print(f"Provider:        {generator.provider}")
        print(f"Requested model: {generator.model}")
        print(f"API reports:     {generator.check_access()}")
        if not args.probe:
            return 0
        call = generator.generate(PROBE)
    except RecommendationConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ProviderError as e:
        kind = "fatal" if e.fatal else "transient" if e.transient else "error"
        print(f"error ({kind}) {e.error_type}: {e.message}", file=sys.stderr)
        return 1

    output, issues = validate_output(call.output_text, PROBE, BRAND_DOMAINS)
    cost = estimate_cost(config.price, call.input_tokens, call.cached_input_tokens, call.output_tokens)
    print(f"Probe model id:  {call.model_id}   latency {call.latency_ms} ms")
    print(f"Tokens:          input {call.input_tokens} (cached {call.cached_input_tokens}), "
          f"output {call.output_tokens} (thinking {call.reasoning_tokens})")
    print(f"Estimated cost:  {'unknown (no price configured)' if cost is None else f'${cost:.6f}'}")
    try:
        print("Output:\n" + json.dumps(json.loads(call.output_text), indent=2))
    except json.JSONDecodeError:
        print(f"Output:          {call.output_text}")
    print("Validation:      " + ("ok" if output else "FAILED: " + "; ".join(issues)))
    return 0 if output else 2


if __name__ == "__main__":
    sys.exit(main())
