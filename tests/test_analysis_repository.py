"""AnalysisRepository against the real database, inside one transaction that is always rolled back."""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
from sqlalchemy import text

from visibility_analysis.repository import AnalysisRepository

pytestmark = [pytest.mark.db,
              pytest.mark.skipif(os.environ.get("RUN_DB_TESTS") != "1", reason="set RUN_DB_TESTS=1 to run")]

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
        with conn.begin_nested():
            yield conn

    return AnalysisRepository(conn.engine, begin)


def extraction(conn, observation_id, status="succeeded", version=VERSION):
    conn.execute(text("""
        INSERT INTO observation_extractions (observation_id, extractor_version, status, validation_status, error_message)
        VALUES (:o, :v, :s, CASE WHEN :s = 'succeeded' THEN 'valid' END, CASE WHEN :s = 'failed' THEN 'test' END)
    """), {"o": observation_id, "v": version, "s": status})


def mention(conn, observation_id, brand_id, recommendation):
    conn.execute(text("""
        INSERT INTO brand_mentions (observation_id, brand_id, extractor_version, mention_count, match_method,
                                    recommendation)
        VALUES (:o, :b, :v, 1, 'alias', :r)
    """), {"o": observation_id, "b": brand_id, "v": VERSION, "r": recommendation})


def citation(conn, observation_id, position, domain, brand_id=None, metadata="{}"):
    conn.execute(text("""
        INSERT INTO citations (observation_id, extractor_version, citation_position, url, domain, brand_id,
                               citation_source, provider_format, metadata)
        VALUES (:o, :v, :p, 'https://' || coalesce(:d, 'x') || '/', :d, :b, 'response_text', 'text_url',
                CAST(:m AS jsonb))
    """), {"o": observation_id, "v": VERSION, "p": position, "d": domain, "b": brand_id, "m": metadata})


def test_load_brands(repo):
    brand_ids, target = repo.load_brands()
    assert target == "hubspot" and "hubspot" in brand_ids and brand_ids == sorted(brand_ids)


def test_load_answers(conn, repo):
    ok, empty, failed, other_version = conn.execute(text(
        "SELECT observation_id FROM observations WHERE status = 'succeeded' ORDER BY observation_id LIMIT 4"
    )).scalars().all()
    for o in (ok, empty):
        extraction(conn, o)
    extraction(conn, failed, status="failed")
    extraction(conn, other_version, version="test-rollback-other")

    mention(conn, ok, "hubspot", "primary")
    mention(conn, ok, "pipedrive", "none")
    mention(conn, ok, "salesforce", None)  # unlabelled counts as not recommended
    citation(conn, ok, 1, "hubspot.com", "hubspot")
    citation(conn, ok, 2, "hubspot.com", "hubspot")
    citation(conn, ok, 3, "g2.com")
    citation(conn, ok, 4, "g2.com")
    citation(conn, ok, 5, "zoho.com", metadata='{"brand_attribution": "skipped: zoho.com is shared"}')
    citation(conn, ok, 6, None)

    answers = {a.observation_id: a for a in repo.load_answers(VERSION)}
    assert set(answers) == {ok, empty}
    a = answers[ok]
    assert a.mentioned == {"hubspot", "pipedrive", "salesforce"}
    assert (a.recommended, a.cited, a.domains) == ({"hubspot"}, {"hubspot"}, {"g2.com"})
    assert not (answers[empty].mentioned | answers[empty].cited | answers[empty].domains)

    query_id, provider, intent = conn.execute(text("""
        SELECT r.query_id, o.provider, q.intent FROM observations o
        JOIN collection_requests r USING (request_id) JOIN queries q USING (query_id)
        WHERE o.observation_id = :o"""), {"o": ok}).one()
    assert (a.query_id, a.provider, a.intent) == (query_id, provider, intent)

    other = "gemini" if provider == "openai" else "openai"
    assert ok not in {x.observation_id for x in repo.load_answers(VERSION, [other])}
