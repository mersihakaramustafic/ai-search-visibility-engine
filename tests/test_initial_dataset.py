"""Tests for scripts/generate_initial_dataset.py and its committed outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

import generate_initial_dataset as gen

PROCESSED_DIR = gen.PROJECT_ROOT / "data" / "processed"
INTENTS = [
    "category_discovery",
    "use_case",
    "brand_comparison",
    "features_integrations",
    "pricing_purchase",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("processed")
    gen.generate(gen.DEFAULT_CONFIG_PATH, out)
    return out


@pytest.fixture(scope="module")
def tables(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {name: read_csv(output_dir / filename) for name, (filename, _) in gen.TABLES.items()}


def write_config(tmp_path: Path, **overrides) -> Path:
    """Copy the default config with top-level section overrides merged in."""
    config = yaml.safe_load(gen.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for section, values in overrides.items():
        config[section].update(values)
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


# --- brands ---------------------------------------------------------------


def test_exactly_ten_unique_brands(tables):
    brands = tables["brands"]
    assert len(brands) == 10
    assert len({b["brand_id"] for b in brands}) == 10
    assert len({b["canonical_domain"] for b in brands}) == 10


def test_single_primary_brand_is_hubspot(tables):
    assert [b["brand_id"] for b in tables["brands"] if b["is_primary_brand"] == "true"] == ["hubspot"]


def test_aliases_are_json_arrays_including_brand_name(tables):
    for brand in tables["brands"]:
        aliases = json.loads(brand["aliases"])
        assert isinstance(aliases, list) and aliases
        assert brand["brand_name"] in aliases


def test_domains_are_bare_hostnames(tables):
    for brand in tables["brands"]:
        assert gen.DOMAIN_RE.match(brand["canonical_domain"]), brand["canonical_domain"]


# --- queries --------------------------------------------------------------


def test_exactly_100_unique_queries(tables):
    queries = tables["queries"]
    assert len(queries) == 100
    assert [q["query_id"] for q in queries] == [f"Q{i:03d}" for i in range(1, 101)]
    assert len({q["query_text"].lower() for q in queries}) == 100


def test_exactly_20_queries_per_intent(tables):
    assert Counter(q["intent"] for q in tables["queries"]) == {intent: 20 for intent in INTENTS}


def test_most_queries_are_brand_neutral_and_identifiable(tables):
    queries = tables["queries"]
    neutral = [q for q in queries if q["contains_brand_name"] == "false"]
    assert len(neutral) / len(queries) >= 0.6
    assert all(json.loads(q["named_brands"]) == [] for q in neutral)


def test_named_brands_reference_known_brands(tables):
    brand_ids = {b["brand_id"] for b in tables["brands"]}
    for query in tables["queries"]:
        named = json.loads(query["named_brands"])
        assert set(named) <= brand_ids
        assert (query["contains_brand_name"] == "true") == bool(named)


def test_queries_do_not_name_providers(tables):
    for query in tables["queries"]:
        text = query["query_text"].lower()
        for term in ("openai", "chatgpt", "gemini"):
            assert term not in text, query["query_id"]


def test_each_query_family_has_one_topic(tables):
    topics: dict[str, set[str]] = {}
    for query in tables["queries"]:
        topics.setdefault(query["query_family_id"], set()).add(query["topic"])
    assert all(len(t) == 1 for t in topics.values())


# --- manifest -------------------------------------------------------------


def test_exactly_600_manifest_rows(tables):
    assert len(tables["collection_manifest"]) == 600


def test_each_query_has_both_providers_and_three_runs(tables):
    plan = Counter((r["query_id"], r["provider"], r["run_number"]) for r in tables["collection_manifest"])
    for query in tables["queries"]:
        for provider in ("openai", "gemini"):
            for run in ("1", "2", "3"):
                assert plan[(query["query_id"], provider, run)] == 1


def test_manifest_query_ids_exist_in_queries(tables):
    query_ids = {q["query_id"] for q in tables["queries"]}
    assert {r["query_id"] for r in tables["collection_manifest"]} <= query_ids


def test_manifest_rows_are_pending_plans(tables):
    for row in tables["collection_manifest"]:
        assert row["status"] == "pending"
        assert row["request_id"] == f"REQ-{row['query_id']}-{row['provider']}-R{row['run_number']}"
    # Planned requests only: no response, model, or timestamp columns.
    assert list(tables["collection_manifest"][0]) == gen.MANIFEST_COLUMNS


# --- recommendation candidates -------------------------------------------


def test_all_candidates_are_synthetic(tables):
    candidates = tables["recommendation_candidates"]
    assert len(candidates) == 30
    assert all(c["is_synthetic"] == "true" for c in candidates)
    assert all(c["hypothesis"].startswith("Untested hypothesis:") for c in candidates)


def test_candidate_action_types_are_balanced(tables):
    counts = Counter(c["action_type"] for c in tables["recommendation_candidates"])
    assert set(counts) == {
        "owned_content_optimization",
        "editorial_outreach",
        "community_visibility",
        "directory_listing",
    }
    assert max(counts.values()) - min(counts.values()) <= 1


def test_candidates_reference_existing_brands_and_families(tables):
    brand_ids = {b["brand_id"] for b in tables["brands"]}
    family_topics = {q["query_family_id"]: q["topic"] for q in tables["queries"]}
    for candidate in tables["recommendation_candidates"]:
        assert candidate["target_brand"] in brand_ids
        assert family_topics[candidate["query_family_id"]] == candidate["topic"]


# --- cross-table integrity ------------------------------------------------


@pytest.mark.parametrize("name", list(gen.TABLES))
def test_no_duplicate_ids_or_missing_values(tables, name):
    _, columns = gen.TABLES[name]
    rows = tables[name]
    id_column = columns[0]
    assert list(rows[0]) == columns
    assert len({r[id_column] for r in rows}) == len(rows)
    for row in rows:
        assert all(row[c].strip() for c in columns), row[id_column]


# --- reproducibility and reruns -------------------------------------------


def test_reproducible_across_runs(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    gen.generate(gen.DEFAULT_CONFIG_PATH, first)
    gen.generate(gen.DEFAULT_CONFIG_PATH, second)
    for filename, _ in gen.TABLES.values():
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_rerun_in_place_does_not_duplicate_rows(tmp_path):
    gen.generate(gen.DEFAULT_CONFIG_PATH, tmp_path)
    gen.generate(gen.DEFAULT_CONFIG_PATH, tmp_path)
    assert len(read_csv(tmp_path / "collection_manifest.csv")) == 600
    assert len(read_csv(tmp_path / "queries.csv")) == 100


def test_committed_outputs_match_generator():
    result = gen.generate(gen.DEFAULT_CONFIG_PATH, PROCESSED_DIR, write=False)
    assert gen.check_files_match(result) == [], "Run scripts/generate_initial_dataset.py to refresh data/processed/"


# --- configuration and failure modes --------------------------------------


def test_config_controls_providers_and_repetitions(tmp_path):
    config = write_config(
        tmp_path,
        collection={
            "repetitions": 5,
            "providers": [
                {"id": "openai", "search_enabled": True},
                {"id": "gemini", "search_enabled": False},
                {"id": "third_provider", "search_enabled": True},
            ],
        },
    )
    result = gen.generate(config, tmp_path / "out")
    manifest = result.tables["collection_manifest"]
    assert len(manifest) == 100 * 3 * 5
    assert {r["search_enabled"] for r in manifest if r["provider"] == "gemini"} == {"false"}


def test_query_subset_must_keep_candidate_families(tmp_path):
    config = write_config(
        tmp_path,
        queries={
            "total": 10,
            "intent_distribution": {intent: 2 for intent in INTENTS},
            "min_brand_neutral_share": 0.0,
            "max_single_brand_share_of_branded": 1.0,
        },
        collection={"repetitions": 2},
        recommendation_candidates={"total": 30},
    )
    # Candidates reference families outside this 10-query subset, so the
    # generator must reject it rather than write dangling references.
    with pytest.raises(gen.DatasetValidationError, match="has no queries in queries.csv"):
        gen.generate(config, tmp_path / "out")


def test_fails_when_too_few_authored_queries(tmp_path):
    distribution = {intent: 20 for intent in INTENTS} | {"use_case": 25}
    config = write_config(tmp_path, queries={"total": 105, "intent_distribution": distribution})
    with pytest.raises(gen.DatasetValidationError, match="only 20 are authored"):
        gen.generate(config, tmp_path / "out")


def test_fails_when_total_mismatches_distribution(tmp_path):
    config = write_config(tmp_path, queries={"total": 99})
    with pytest.raises(gen.DatasetValidationError, match="does not equal the sum"):
        gen.generate(config, tmp_path / "out")


def test_refuses_to_reset_progressed_manifest(tmp_path):
    gen.generate(gen.DEFAULT_CONFIG_PATH, tmp_path)
    manifest = tmp_path / "collection_manifest.csv"
    manifest.write_text(manifest.read_text().replace("pending", "completed", 1))
    with pytest.raises(gen.DatasetValidationError, match="--force"):
        gen.generate(gen.DEFAULT_CONFIG_PATH, tmp_path)
    gen.generate(gen.DEFAULT_CONFIG_PATH, tmp_path, force=True)
    assert "completed" not in manifest.read_text()


def test_cli_exits_nonzero_with_message_on_failure(tmp_path, capsys):
    config = write_config(tmp_path, queries={"total": 99})
    assert gen.main(["--config", str(config), "--output-dir", str(tmp_path / "out")]) == 1
    assert "Dataset validation failed" in capsys.readouterr().err
