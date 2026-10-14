#!/usr/bin/env python3
"""Generate the initial datasets for the AI search visibility engine.

Builds four CSV tables from configs/dataset.yaml and the hand-written content
in data/authored/, validates them, and writes them to data/processed/:

    brands.csv                     authored brand metadata
    queries.csv                    authored prompts with derived brand fields
    collection_manifest.csv        planned API requests (no responses)
    recommendation_candidates.csv  SYNTHETIC illustrative seed rows

The pipeline is deterministic and offline: no API keys, network access, or
LLM calls. Re-running overwrites each file atomically, so rows never
duplicate.

Usage:
    python scripts/generate_initial_dataset.py [--config PATH] [--output-dir PATH]
                                               [--check] [--force]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset.yaml"

BRAND_COLUMNS = [
    "brand_id",
    "brand_name",
    "canonical_domain",
    "aliases",
    "description",
    "product_category",
    "is_primary_brand",
]
QUERY_COLUMNS = [
    "query_id",
    "query_text",
    "intent",
    "topic",
    "target_audience",
    "language",
    "market",
    "contains_brand_name",
    "named_brands",
    "query_family_id",
]
MANIFEST_COLUMNS = [
    "request_id",
    "query_id",
    "provider",
    "run_number",
    "market",
    "language",
    "search_enabled",
    "status",
]
CANDIDATE_COLUMNS = [
    "candidate_id",
    "target_brand",
    "action_type",
    "topic",
    "query_family_id",
    "evidence_pattern",
    "suggested_action",
    "evidence_required",
    "estimated_effort",
    "hypothesis",
    "is_synthetic",
]

TABLES = {
    "brands": ("brands.csv", BRAND_COLUMNS),
    "queries": ("queries.csv", QUERY_COLUMNS),
    "collection_manifest": ("collection_manifest.csv", MANIFEST_COLUMNS),
    "recommendation_candidates": ("recommendation_candidates.csv", CANDIDATE_COLUMNS),
}

# Lowercase hostname without scheme, path, port, or trailing dot.
DOMAIN_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
BRAND_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
QUERY_ID_RE = re.compile(r"^Q\d{3}$")
FAMILY_ID_RE = re.compile(r"^fam_[a-z0-9_]+$")
CANDIDATE_ID_RE = re.compile(r"^REC\d{3}$")
PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TOKEN_RE = re.compile(r"[a-z0-9$']+")

Row = dict[str, str]


class DatasetValidationError(Exception):
    """Raised when generated data fails validation. Lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"Dataset validation failed with {len(errors)} error(s):\n{bullet_list}")


@dataclass
class GenerationResult:
    tables: dict[str, list[Row]]
    csv_text: dict[str, str]
    output_dir: Path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetValidationError([f"File not found: {path}"])
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise DatasetValidationError([f"Expected a YAML mapping in {path}"])
    return data


def load_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    required = {
        "market": str,
        "language": str,
        "paths": dict,
        "brands": dict,
        "queries": dict,
        "collection": dict,
        "recommendation_candidates": dict,
    }
    errors = [
        f"Config {path}: missing or invalid '{key}' (expected {kind.__name__})"
        for key, kind in required.items()
        if not isinstance(config.get(key), kind)
    ]
    if errors:
        raise DatasetValidationError(errors)

    distribution = config["queries"].get("intent_distribution") or {}
    if sum(distribution.values()) != config["queries"].get("total"):
        errors.append(
            f"Config: queries.total ({config['queries'].get('total')}) does not equal the sum of "
            f"queries.intent_distribution ({sum(distribution.values())})"
        )
    if any(not isinstance(n, int) or n < 0 for n in distribution.values()):
        errors.append("Config: intent_distribution counts must be non-negative integers")

    collection = config["collection"]
    if not isinstance(collection.get("repetitions"), int) or collection["repetitions"] < 1:
        errors.append("Config: collection.repetitions must be a positive integer")
    providers = collection.get("providers") or []
    provider_ids = [p.get("id") for p in providers if isinstance(p, dict)]
    if not providers or len(provider_ids) != len(providers):
        errors.append("Config: collection.providers must be a non-empty list of {id, search_enabled}")
    if len(set(provider_ids)) != len(provider_ids):
        errors.append("Config: duplicate provider ids in collection.providers")
    for provider in providers:
        if not PROVIDER_ID_RE.match(str(provider.get("id", ""))):
            errors.append(f"Config: invalid provider id {provider.get('id')!r}")
        if not isinstance(provider.get("search_enabled"), bool):
            errors.append(f"Config: provider {provider.get('id')!r} needs a boolean search_enabled")
    if errors:
        raise DatasetValidationError(errors)
    return config


# --------------------------------------------------------------------------
# Serialization helpers
# --------------------------------------------------------------------------


def to_bool_str(value: bool) -> str:
    return "true" if value else "false"


def to_json_list(values: list[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def parse_json_list(value: str) -> list[str] | None:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list) or not all(isinstance(v, str) for v in parsed):
        return None
    return parsed


def rows_to_csv(columns: list[str], rows: list[Row]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_atomically(path: Path, text: str) -> None:
    """Replace path with text in one step, so an interrupted run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def build_brands(authored: dict[str, Any]) -> list[Row]:
    rows = []
    for brand in authored.get("brands", []):
        rows.append(
            {
                "brand_id": str(brand.get("brand_id", "")).strip(),
                "brand_name": str(brand.get("brand_name", "")).strip(),
                "canonical_domain": str(brand.get("canonical_domain", "")).strip(),
                "aliases": to_json_list([str(a).strip() for a in brand.get("aliases", [])]),
                "description": " ".join(str(brand.get("description", "")).split()),
                "product_category": str(brand.get("product_category", "")).strip(),
                "is_primary_brand": to_bool_str(brand.get("is_primary_brand") is True),
            }
        )
    return sorted(rows, key=lambda r: r["brand_id"])


def compile_alias_patterns(brands: list[Row]) -> dict[str, list[re.Pattern[str]]]:
    """Whole-word, case-sensitive alias patterns per brand (longest alias first)."""
    patterns = {}
    for brand in brands:
        aliases = sorted(parse_json_list(brand["aliases"]) or [], key=len, reverse=True)
        patterns[brand["brand_id"]] = [
            re.compile(rf"(?<![\w.]){re.escape(alias)}(?![\w])") for alias in aliases if alias
        ]
    return patterns


def detect_named_brands(text: str, alias_patterns: dict[str, list[re.Pattern[str]]]) -> list[str]:
    found = {brand_id for brand_id, pats in alias_patterns.items() if any(p.search(text) for p in pats)}
    return sorted(found)


def build_queries(
    authored: dict[str, Any], brands: list[Row], config: dict[str, Any]
) -> tuple[list[Row], list[str]]:
    """Select the configured number of queries per intent and derive brand fields.

    Returns (rows, errors) so selection problems are reported with all others.
    """
    errors: list[str] = []
    families: dict[str, str] = authored.get("families") or {}
    distribution: dict[str, int] = config["queries"]["intent_distribution"]
    alias_patterns = compile_alias_patterns(brands)

    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in authored.get("queries", []):
        by_intent[str(query.get("intent", ""))].append(query)

    unknown_intents = sorted(set(by_intent) - set(distribution))
    if unknown_intents:
        errors.append(f"Authored queries use intents not in config: {unknown_intents}")

    rows = []
    for intent, wanted in distribution.items():
        available = by_intent.get(intent, [])
        if len(available) < wanted:
            errors.append(
                f"Intent '{intent}': config asks for {wanted} queries but only "
                f"{len(available)} are authored in queries.yaml"
            )
        for query in available[:wanted]:
            text = " ".join(str(query.get("text", "")).split())
            family_id = str(query.get("family", "")).strip()
            named = detect_named_brands(text, alias_patterns)
            rows.append(
                {
                    "query_id": str(query.get("id", "")).strip(),
                    "query_text": text,
                    "intent": intent,
                    "topic": families.get(family_id, ""),
                    "target_audience": str(query.get("audience", "")).strip(),
                    "language": config["language"],
                    "market": config["market"],
                    "contains_brand_name": to_bool_str(bool(named)),
                    "named_brands": to_json_list(named),
                    "query_family_id": family_id,
                }
            )
    return sorted(rows, key=lambda r: r["query_id"]), errors


def make_request_id(query_id: str, provider: str, run_number: int) -> str:
    return f"REQ-{query_id}-{provider}-R{run_number}"


def build_manifest(queries: list[Row], config: dict[str, Any]) -> list[Row]:
    collection = config["collection"]
    rows = []
    for query in queries:
        for provider in collection["providers"]:
            for run_number in range(1, collection["repetitions"] + 1):
                rows.append(
                    {
                        "request_id": make_request_id(query["query_id"], provider["id"], run_number),
                        "query_id": query["query_id"],
                        "provider": provider["id"],
                        "run_number": str(run_number),
                        "market": query["market"],
                        "language": query["language"],
                        "search_enabled": to_bool_str(provider["search_enabled"]),
                        "status": collection["initial_status"],
                    }
                )
    return rows


def build_candidates(authored: dict[str, Any], families: dict[str, str]) -> list[Row]:
    rows = []
    for candidate in authored.get("candidates", []):
        family_id = str(candidate.get("family", "")).strip()
        rows.append(
            {
                "candidate_id": str(candidate.get("id", "")).strip(),
                "target_brand": str(candidate.get("target_brand", "")).strip(),
                "action_type": str(candidate.get("action_type", "")).strip(),
                "topic": families.get(family_id, ""),
                "query_family_id": family_id,
                "evidence_pattern": str(candidate.get("evidence_pattern", "")).strip(),
                "suggested_action": " ".join(str(candidate.get("suggested_action", "")).split()),
                "evidence_required": " ".join(str(candidate.get("evidence_required", "")).split()),
                "estimated_effort": str(candidate.get("effort", "")).strip(),
                "hypothesis": " ".join(str(candidate.get("hypothesis", "")).split()),
                # Always synthetic: this seed file never contains observed recommendations.
                "is_synthetic": "true",
            }
        )
    return sorted(rows, key=lambda r: r["candidate_id"])


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def check_common(name: str, rows: list[Row], columns: list[str], id_column: str) -> list[str]:
    errors = []
    for index, row in enumerate(rows):
        if list(row) != columns:
            errors.append(f"{name}: row {index} has columns {list(row)}, expected {columns}")
            continue
        missing = [c for c in columns if not str(row[c]).strip()]
        if missing:
            errors.append(f"{name}: {row.get(id_column) or f'row {index}'} has empty required field(s) {missing}")
    duplicates = sorted(k for k, n in Counter(r.get(id_column) for r in rows).items() if n > 1)
    if duplicates:
        errors.append(f"{name}: duplicate {id_column} values {duplicates}")
    return errors


def validate_brands(brands: list[Row], authored: dict[str, Any], config: dict[str, Any]) -> list[str]:
    errors = check_common("brands", brands, BRAND_COLUMNS, "brand_id")
    expected = config["brands"]["expected_count"]
    if len(brands) != expected:
        errors.append(f"brands: expected {expected} brands, found {len(brands)}")

    categories = set(authored.get("product_categories") or [])
    alias_owner: dict[str, str] = {}
    for brand in brands:
        brand_id = brand["brand_id"]
        if not BRAND_ID_RE.match(brand_id):
            errors.append(f"brands: invalid brand_id {brand_id!r} (use lowercase snake_case)")
        if not DOMAIN_RE.match(brand["canonical_domain"]):
            errors.append(
                f"brands: {brand_id} canonical_domain {brand['canonical_domain']!r} is not a bare "
                "lowercase hostname (no scheme, path, or port)"
            )
        if brand["product_category"] not in categories:
            errors.append(f"brands: {brand_id} product_category {brand['product_category']!r} not in {sorted(categories)}")
        aliases = parse_json_list(brand["aliases"])
        if not aliases:
            errors.append(f"brands: {brand_id} aliases must be a non-empty JSON array of strings")
            continue
        if brand["brand_name"] not in aliases:
            errors.append(f"brands: {brand_id} aliases must include brand_name {brand['brand_name']!r}")
        if len(set(aliases)) != len(aliases) or any(not a.strip() for a in aliases):
            errors.append(f"brands: {brand_id} has duplicate or empty aliases")
        for alias in aliases:
            owner = alias_owner.setdefault(alias.casefold(), brand_id)
            if owner != brand_id:
                errors.append(f"brands: alias {alias!r} is shared by {owner} and {brand_id}")

    domains = Counter(b["canonical_domain"] for b in brands)
    errors += [f"brands: canonical_domain {d!r} used by multiple brands" for d, n in domains.items() if n > 1]

    primary = [b["brand_id"] for b in brands if b["is_primary_brand"] == "true"]
    if primary != [config["brands"]["primary_brand_id"]]:
        errors.append(
            f"brands: expected exactly one primary brand "
            f"({config['brands']['primary_brand_id']!r}), found {primary}"
        )
    return errors


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def validate_queries(
    queries: list[Row], brands: list[Row], authored: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    errors = check_common("queries", queries, QUERY_COLUMNS, "query_id")
    settings = config["queries"]
    distribution = settings["intent_distribution"]
    families: dict[str, str] = authored.get("families") or {}
    audiences = set(authored.get("target_audiences") or [])
    brand_ids = {b["brand_id"] for b in brands}

    if len(queries) != settings["total"]:
        errors.append(f"queries: expected {settings['total']} queries, found {len(queries)}")
    intent_counts = Counter(q["intent"] for q in queries)
    for intent, wanted in distribution.items():
        if intent_counts.get(intent, 0) != wanted:
            errors.append(f"queries: intent '{intent}' has {intent_counts.get(intent, 0)} queries, expected {wanted}")

    forbidden = [
        (term, re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE))
        for term in settings.get("forbidden_terms", [])
    ]
    for query in queries:
        qid = query["query_id"]
        if not QUERY_ID_RE.match(qid):
            errors.append(f"queries: invalid query_id {qid!r} (expected Q followed by 3 digits)")
        if query["intent"] not in distribution:
            errors.append(f"queries: {qid} has unknown intent {query['intent']!r}")
        if query["query_family_id"] not in families or not FAMILY_ID_RE.match(query["query_family_id"]):
            errors.append(f"queries: {qid} has undefined query_family_id {query['query_family_id']!r}")
        if query["target_audience"] not in audiences:
            errors.append(f"queries: {qid} has unknown target_audience {query['target_audience']!r}")
        if query["language"] != config["language"] or query["market"] != config["market"]:
            errors.append(f"queries: {qid} language/market do not match config")
        if not query["query_text"].endswith("?"):
            errors.append(f"queries: {qid} should be phrased as a question ending in '?'")
        for term, pattern in forbidden:
            if pattern.search(query["query_text"]):
                errors.append(f"queries: {qid} mentions forbidden provider term {term!r}")
        named = parse_json_list(query["named_brands"])
        if named is None:
            errors.append(f"queries: {qid} named_brands is not a JSON array of strings")
            continue
        if set(named) - brand_ids:
            errors.append(f"queries: {qid} named_brands references unknown brands {sorted(set(named) - brand_ids)}")
        if query["contains_brand_name"] != to_bool_str(bool(named)):
            errors.append(f"queries: {qid} contains_brand_name disagrees with named_brands")

    # Duplicates and near-duplicates (superficial rewordings).
    normalized = Counter(" ".join(TOKEN_RE.findall(q["query_text"].lower())) for q in queries)
    errors += [f"queries: duplicate query text {t!r}" for t, n in normalized.items() if n > 1]
    threshold = settings["near_duplicate_jaccard_threshold"]
    tokens = [(q["query_id"], set(TOKEN_RE.findall(q["query_text"].lower()))) for q in queries]
    for i, (id_a, tok_a) in enumerate(tokens):
        for id_b, tok_b in tokens[i + 1 :]:
            score = jaccard(tok_a, tok_b)
            if score >= threshold:
                errors.append(f"queries: {id_a} and {id_b} look like near-duplicates (token Jaccard {score:.2f})")

    # Brand balance: most prompts neutral, no single brand dominating branded prompts.
    branded = [q for q in queries if q["contains_brand_name"] == "true"]
    if queries:
        neutral_share = 1 - len(branded) / len(queries)
        if neutral_share < settings["min_brand_neutral_share"]:
            errors.append(
                f"queries: brand-neutral share {neutral_share:.2f} is below "
                f"min_brand_neutral_share {settings['min_brand_neutral_share']}"
            )
    if branded:
        mentions = Counter(b for q in branded for b in parse_json_list(q["named_brands"]) or [])
        for brand_id, count in mentions.items():
            share = count / len(branded)
            if share > settings["max_single_brand_share_of_branded"]:
                errors.append(
                    f"queries: {brand_id} is named in {share:.0%} of branded queries, above "
                    f"max_single_brand_share_of_branded {settings['max_single_brand_share_of_branded']:.0%}"
                )

    # Query families: one topic per family, bounded size.
    family_topics: dict[str, set[str]] = defaultdict(set)
    for query in queries:
        family_topics[query["query_family_id"]].add(query["topic"])
    for family_id, topics in family_topics.items():
        if len(topics) != 1:
            errors.append(f"queries: family {family_id} spans multiple topics {sorted(topics)}")
    family_sizes = Counter(q["query_family_id"] for q in queries)
    for family_id, size in family_sizes.items():
        if size > settings["max_family_size"]:
            errors.append(f"queries: family {family_id} has {size} queries, above max_family_size {settings['max_family_size']}")
    return errors


def validate_manifest(manifest: list[Row], queries: list[Row], config: dict[str, Any]) -> list[str]:
    errors = check_common("collection_manifest", manifest, MANIFEST_COLUMNS, "request_id")
    collection = config["collection"]
    providers = {p["id"]: p for p in collection["providers"]}
    repetitions = collection["repetitions"]
    query_by_id = {q["query_id"]: q for q in queries}

    expected = len(queries) * len(providers) * repetitions
    if len(manifest) != expected:
        errors.append(
            f"collection_manifest: expected {expected} rows "
            f"({len(queries)} queries x {len(providers)} providers x {repetitions} runs), found {len(manifest)}"
        )

    seen = Counter((r["query_id"], r["provider"], r["run_number"]) for r in manifest)
    errors += [f"collection_manifest: duplicate plan entry {key}" for key, n in seen.items() if n > 1]
    for query_id in query_by_id:
        for provider in providers:
            for run in range(1, repetitions + 1):
                if (query_id, provider, str(run)) not in seen:
                    errors.append(f"collection_manifest: missing {query_id} / {provider} / run {run}")

    for row in manifest:
        rid = row["request_id"]
        query = query_by_id.get(row["query_id"])
        if query is None:
            errors.append(f"collection_manifest: {rid} references unknown query_id {row['query_id']!r}")
            continue
        provider = providers.get(row["provider"])
        if provider is None:
            errors.append(f"collection_manifest: {rid} has unknown provider {row['provider']!r}")
            continue
        if not row["run_number"].isdigit() or not 1 <= int(row["run_number"]) <= repetitions:
            errors.append(f"collection_manifest: {rid} run_number {row['run_number']!r} outside 1..{repetitions}")
            continue
        if rid != make_request_id(row["query_id"], row["provider"], int(row["run_number"])):
            errors.append(f"collection_manifest: {rid} does not follow the stable request_id format")
        if row["market"] != query["market"] or row["language"] != query["language"]:
            errors.append(f"collection_manifest: {rid} market/language differ from its query")
        if row["search_enabled"] != to_bool_str(provider["search_enabled"]):
            errors.append(f"collection_manifest: {rid} search_enabled differs from provider config")
        if row["status"] != collection["initial_status"]:
            errors.append(f"collection_manifest: {rid} status {row['status']!r} is not {collection['initial_status']!r}")
    return errors


def validate_candidates(
    candidates: list[Row], brands: list[Row], queries: list[Row], authored: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    errors = check_common("recommendation_candidates", candidates, CANDIDATE_COLUMNS, "candidate_id")
    settings = config["recommendation_candidates"]
    brand_ids = {b["brand_id"] for b in brands}
    families: dict[str, str] = authored.get("families") or {}
    query_families = {q["query_family_id"] for q in queries}

    if len(candidates) != settings["total"]:
        errors.append(f"recommendation_candidates: expected {settings['total']} rows, found {len(candidates)}")

    action_counts = Counter(c["action_type"] for c in candidates)
    counts = [action_counts.get(a, 0) for a in settings["action_types"]]
    if counts and max(counts) - min(counts) > settings["max_action_type_imbalance"]:
        errors.append(
            f"recommendation_candidates: action types are unbalanced "
            f"{dict(zip(settings['action_types'], counts))}"
        )

    claim_phrases = [p.lower() for p in settings.get("forbidden_claim_phrases", [])]
    for row in candidates:
        cid = row["candidate_id"]
        if not CANDIDATE_ID_RE.match(cid):
            errors.append(f"recommendation_candidates: invalid candidate_id {cid!r}")
        if row["target_brand"] not in brand_ids:
            errors.append(f"recommendation_candidates: {cid} target_brand {row['target_brand']!r} is not a known brand_id")
        if row["action_type"] not in settings["action_types"]:
            errors.append(f"recommendation_candidates: {cid} unknown action_type {row['action_type']!r}")
        if row["evidence_pattern"] not in settings["evidence_patterns"]:
            errors.append(f"recommendation_candidates: {cid} unknown evidence_pattern {row['evidence_pattern']!r}")
        if row["estimated_effort"] not in settings["effort_levels"]:
            errors.append(f"recommendation_candidates: {cid} unknown estimated_effort {row['estimated_effort']!r}")
        if row["query_family_id"] not in query_families:
            errors.append(
                f"recommendation_candidates: {cid} query_family_id {row['query_family_id']!r} "
                "has no queries in queries.csv"
            )
        elif row["topic"] != families.get(row["query_family_id"]):
            errors.append(f"recommendation_candidates: {cid} topic does not match its query family")
        if row["is_synthetic"] != "true":
            errors.append(f"recommendation_candidates: {cid} must be marked is_synthetic=true")
        if not row["hypothesis"].startswith("Untested hypothesis:"):
            errors.append(f"recommendation_candidates: {cid} hypothesis must start with 'Untested hypothesis:'")
        text = " ".join(row[c] for c in ("suggested_action", "evidence_required", "hypothesis")).lower()
        for phrase in claim_phrases:
            if phrase in text:
                errors.append(f"recommendation_candidates: {cid} overstates evidence with {phrase!r}")
    return errors


def validate_all(tables: dict[str, list[Row]], authored: dict[str, dict[str, Any]], config: dict[str, Any]) -> list[str]:
    return (
        validate_brands(tables["brands"], authored["brands"], config)
        + validate_queries(tables["queries"], tables["brands"], authored["queries"], config)
        + validate_manifest(tables["collection_manifest"], tables["queries"], config)
        + validate_candidates(
            tables["recommendation_candidates"], tables["brands"], tables["queries"], authored["queries"], config
        )
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_tables(config: dict[str, Any]) -> tuple[dict[str, list[Row]], list[str]]:
    authored_dir = resolve_path(config["paths"]["authored_dir"])
    authored = {
        "brands": load_yaml(authored_dir / "brands.yaml"),
        "queries": load_yaml(authored_dir / "queries.yaml"),
        "candidates": load_yaml(authored_dir / "recommendation_candidates.yaml"),
    }
    brands = build_brands(authored["brands"])
    queries, selection_errors = build_queries(authored["queries"], brands, config)
    tables = {
        "brands": brands,
        "queries": queries,
        "collection_manifest": build_manifest(queries, config),
        "recommendation_candidates": build_candidates(authored["candidates"], authored["queries"].get("families") or {}),
    }
    errors = selection_errors + validate_all(tables, authored, config)
    return tables, errors


def guard_existing_manifest(output_dir: Path, initial_status: str) -> None:
    """Refuse to reset a manifest whose statuses were already advanced by a collector."""
    path = output_dir / TABLES["collection_manifest"][0]
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        progressed = [r["request_id"] for r in csv.DictReader(handle) if r.get("status") != initial_status]
    if progressed:
        raise DatasetValidationError(
            [
                f"{path} has {len(progressed)} request(s) with a status other than {initial_status!r} "
                f"(e.g. {progressed[0]}). Regenerating would reset them. Re-run with --force to overwrite."
            ]
        )


def generate(
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path | None = None,
    *,
    write: bool = True,
    force: bool = False,
) -> GenerationResult:
    config = load_config(config_path)
    out_dir = output_dir or resolve_path(config["paths"]["output_dir"])
    tables, errors = build_tables(config)
    if errors:
        raise DatasetValidationError(errors)

    csv_text = {name: rows_to_csv(TABLES[name][1], rows) for name, rows in tables.items()}
    if write:
        if not force:
            guard_existing_manifest(out_dir, config["collection"]["initial_status"])
        for name, text in csv_text.items():
            write_atomically(out_dir / TABLES[name][0], text)
    return GenerationResult(tables=tables, csv_text=csv_text, output_dir=out_dir)


def check_files_match(result: GenerationResult) -> list[str]:
    """Return the files on disk that differ from freshly generated content."""
    stale = []
    for name, text in result.csv_text.items():
        path = result.output_dir / TABLES[name][0]
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            stale.append(str(path))
    return stale


def print_summary(result: GenerationResult, wrote: bool) -> None:
    tables = result.tables
    queries = tables["queries"]
    branded = sum(q["contains_brand_name"] == "true" for q in queries)
    families = {q["query_family_id"] for q in queries}
    intents = Counter(q["intent"] for q in queries)
    providers = Counter(r["provider"] for r in tables["collection_manifest"])
    actions = Counter(c["action_type"] for c in tables["recommendation_candidates"])

    print("Dataset generation summary")
    print(f"  brands.csv                     {len(tables['brands']):>4} rows")
    print(f"  queries.csv                    {len(queries):>4} rows  "
          f"({len(queries) - branded} brand-neutral, {branded} branded, {len(families)} families)")
    print("    intents: " + ", ".join(f"{k}={v}" for k, v in intents.items()))
    print(f"  collection_manifest.csv        {len(tables['collection_manifest']):>4} rows  "
          f"({', '.join(f'{k}={v}' for k, v in providers.items())})")
    print(f"  recommendation_candidates.csv  {len(tables['recommendation_candidates']):>4} rows  (all synthetic)")
    print("    action types: " + ", ".join(f"{k}={v}" for k, v in actions.items()))
    print("Validation: passed")
    if wrote:
        print(f"Wrote CSV files to {result.output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to dataset YAML config")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override paths.output_dir from the config")
    parser.add_argument("--check", action="store_true", help="Validate and verify files on disk are up to date; write nothing")
    parser.add_argument("--force", action="store_true", help="Overwrite a manifest whose statuses were already advanced")
    args = parser.parse_args(argv)

    try:
        result = generate(args.config, args.output_dir, write=not args.check, force=args.force)
    except DatasetValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print_summary(result, wrote=not args.check)
    if args.check:
        stale = check_files_match(result)
        if stale:
            print("Out of date (re-run without --check): " + ", ".join(stale), file=sys.stderr)
            return 1
        print("Files on disk match the generated output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
