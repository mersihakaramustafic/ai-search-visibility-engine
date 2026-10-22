"""ExtractionRepository against the real database, inside one transaction that is always rolled back.

Skipped unless RUN_DB_TESTS=1. Needs DATABASE_URL (or .env) and migrations at head.
Nothing is committed: every statement runs on a single connection whose
transaction is rolled back at the end of each test.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace

import pytest
from sqlalchemy import text

from visibility_extractor.config import load_config
from visibility_extractor.pipeline import LLMRun, RulesExtractor
from visibility_extractor.repository import ExtractionRepository

pytestmark = pytest.mark.skipif(os.environ.get("RUN_DB_TESTS") != "1", reason="set RUN_DB_TESTS=1 to run")

VERSION = "test-rollback"


@pytest.fixture
def conn():
    from visibility_db.config import make_engine

    engine = make_engine()
    with engine.connect() as c:
        tx = c.begin()
        try:
            yield c
        finally:
            tx.rollback()
    engine.dispose()


@pytest.fixture
def repo(conn):
    @contextmanager
    def begin():
        with conn.begin_nested():  # savepoint inside the outer, rolled-back transaction
            yield conn

    return ExtractionRepository(conn.engine, begin)


def counts(conn, observation_id):
    return tuple(conn.execute(text(f"SELECT count(*) FROM {t} WHERE observation_id = :o AND extractor_version = :v"),
                              {"o": observation_id, "v": VERSION}).scalar_one()
                 for t in ("observation_extractions", "brand_mentions", "citations"))


def test_save_is_idempotent_and_reprocess_replaces(conn, repo):
    config = load_config()
    extractor = RulesExtractor(repo.load_brands(), config)
    extractor.version = VERSION

    [obs] = repo.select_observations(VERSION, ["openai"], limit=1)
    extraction = extractor.extract(obs)
    assert extraction.validation_status in ("valid", "valid_with_warnings")

    repo.save(extraction)
    first = counts(conn, obs.observation_id)
    assert first == (1, len(extraction.mentions), len(extraction.citations))
    assert repo.select_observations(VERSION, observation_ids=[obs.observation_id]) == []
    assert repo.select_observations(VERSION, observation_ids=[obs.observation_id], reprocess=True) == [obs]

    repo.save(extraction)  # same version again: replaced, not duplicated
    assert counts(conn, obs.observation_id) == first


def test_failed_extraction_is_recorded(conn, repo):
    extractor = RulesExtractor(repo.load_brands(), load_config())
    extractor.version = VERSION
    [obs] = repo.select_observations(VERSION, ["gemini"], limit=1)
    failed = extractor.extract(obs.__class__(obs.observation_id, obs.provider, None, None))
    repo.save(failed)
    row = conn.execute(text("SELECT status, validation_status, error_type FROM observation_extractions "
                            "WHERE observation_id = :o AND extractor_version = :v"),
                       {"o": obs.observation_id, "v": VERSION}).one()
    assert tuple(row) == ("failed", None, "TypeError")


def test_llm_labels_and_provenance_are_stored(conn, repo):
    extractor = RulesExtractor(repo.load_brands(), load_config())
    extractor.version = VERSION
    [obs] = repo.select_observations(VERSION, ["openai"], limit=1)
    rules = extractor.extract(obs)
    labelled = [replace(m, sentiment="positive", recommendation="conditional", classification_evidence="quote")
                for m in rules.mentions]
    run = LLMRun("succeeded", "gpt-5.6-luna", "p1", "s1", model_id="gpt-5.6-luna",
                 output={"brands": [], "candidates": []}, attempts=2, input_tokens=1000, cached_input_tokens=800,
                 output_tokens=200, reasoning_tokens=50, estimated_cost_usd=0.000296, latency_ms=900)
    repo.save(replace(rules, strategy="rules_and_llm", llm=run, mentions=labelled, target_brand_recommended=True))

    row = conn.execute(text("SELECT strategy, classification_status, classification_attempts, classification_cached_input_tokens, "
                            "classification_estimated_cost_usd, "
                            "classification_output, target_brand_recommended FROM observation_extractions "
                            "WHERE observation_id = :o AND extractor_version = :v"),
                       {"o": obs.observation_id, "v": VERSION}).one()
    assert (row.strategy, row.classification_status, row.classification_attempts,
            row.classification_cached_input_tokens) == (
        "rules_and_llm", "succeeded", 2, 800)
    assert float(row.classification_estimated_cost_usd) == 0.000296
    assert row.classification_output == {"brands": [], "candidates": []}
    labels = conn.execute(text("SELECT DISTINCT sentiment, recommendation FROM brand_mentions "
                               "WHERE observation_id = :o AND extractor_version = :v"),
                          {"o": obs.observation_id, "v": VERSION}).all()
    assert labels == [("positive", "conditional")]
