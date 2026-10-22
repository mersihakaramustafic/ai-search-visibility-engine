#!/usr/bin/env python3
"""Check that the configured extraction model is usable. Never switches to another model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visibility_collector.providers.base import ProviderError  # noqa: E402
from visibility_extractor.config import DEFAULT_CONFIG_PATH, ExtractionConfigError, load_config  # noqa: E402
from visibility_extractor.llm import ClassificationRequest, LLMClassifier, estimate_cost, validate_output  # noqa: E402

PROBE_TEXT = ("For a small team, HubSpot is the best choice. Close is good if you make many calls. "
              "Remember to close deals before Monday.")
PROBE = ClassificationRequest(
    observation_id=0,
    query_text="Which CRM should a small sales team use?",
    response_text=PROBE_TEXT,
    brands=[{"brand_id": "hubspot", "name": "HubSpot", "aliases": ["HubSpot"]},
            {"brand_id": "close", "name": "Close", "aliases": ["Close", "Close CRM"]}],
    candidates=[{"candidate_id": 1, "brand_id": "close", "text": "Close", "char_offset": PROBE_TEXT.index("Close"),
                 "context": "best choice. Close is good if"}],
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--probe", action="store_true", help="Also make one tiny paid classification call")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if config.llm is None:
            raise ExtractionConfigError([f"no llm section in {args.config}"])
        classifier = LLMClassifier.from_config(config.llm)
        print(f"Requested model: {classifier.model}")
        print(f"API reports:     {classifier.check_access()}")
        if not args.probe:
            return 0
        call = classifier.classify(PROBE)
    except ExtractionConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ProviderError as e:
        kind = "fatal" if e.fatal else "transient" if e.transient else "error"
        print(f"error ({kind}) {e.error_type}: {e.message}", file=sys.stderr)
        return 1

    output, issues = validate_output(call.output_text, PROBE, classifier.schema)
    cost = estimate_cost(config.pricing.get(classifier.model), call.input_tokens, call.cached_input_tokens,
                         call.output_tokens)
    print(f"Probe model id:  {call.model_id}   latency {call.latency_ms} ms")
    print(f"Tokens:          input {call.input_tokens} (cached {call.cached_input_tokens}), "
          f"output {call.output_tokens} (reasoning {call.reasoning_tokens})")
    print(f"Estimated cost:  {'unknown (no price configured)' if cost is None else f'${cost:.6f}'}")
    print(f"Output:          {call.output_text}")
    print("Validation:      " + ("ok" if output else "FAILED: " + "; ".join(issues)))
    return 0 if output else 2


if __name__ == "__main__":
    sys.exit(main())
