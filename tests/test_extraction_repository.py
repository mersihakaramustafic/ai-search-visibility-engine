"""ExtractionRepository against the real database, inside one transaction that is always rolled back.

Skipped unless RUN_DB_TESTS=1. Needs DATABASE_URL (or .env) and migrations at head.
Nothing is committed: every statement runs on a single connection whose
transaction is rolled back at the end of each test.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
from sqlalchemy import text

from visibility_extractor.config import load_config
from visibility_extractor.pipeline import RulesExtractor
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
