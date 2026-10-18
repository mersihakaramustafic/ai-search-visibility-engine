"""Validation of configs/collection.yaml."""

from __future__ import annotations

import copy

import pytest

from visibility_collector.config import (
    DEFAULT_CONFIG_PATH,
    CollectorConfigError,
    load_config,
    parse_config,
)

VALID = {
    "providers": {
        "openai": {"model": "gpt-x", "api_key_env": "OPENAI_API_KEY", "timeout_seconds": 60,
                   "max_output_tokens": 1000, "concurrency": 2, "web_search": {"use_market_location": True}},
        "gemini": {"model": "gemini-x", "api_key_env": "GEMINI_API_KEY", "timeout_seconds": 60,
                   "max_output_tokens": 1000, "concurrency": 1},
    },
    "retry": {"max_attempts": 3, "initial_backoff_seconds": 1, "max_backoff_seconds": 10},
    "pilot": {"query_ids": ["Q001", "Q002", "Q003"]},
}


def with_changes(**paths):
    raw = copy.deepcopy(VALID)
    for path, value in paths.items():
        *parents, key = path.split("__")
        node = raw
        for p in parents:
            node = node[p]
        node[key] = value
    return raw


def errors_for(raw) -> list[str]:
    with pytest.raises(CollectorConfigError) as exc:
        parse_config(raw)
    return exc.value.errors


def test_committed_config_is_valid():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert set(config.providers) == {"openai", "gemini"}
    assert 3 <= len(config.pilot_query_ids) <= 5


def test_valid_config_parses():
    config = parse_config(VALID)
    assert config.providers["openai"].model == "gpt-x"
    assert config.providers["openai"].options == {"web_search": {"use_market_location": True}}
    assert config.retry.max_attempts == 3
    assert config.pilot_query_ids == ("Q001", "Q002", "Q003")


@pytest.mark.parametrize("changes, message", [
    ({"providers__openai__model": ""}, "providers.openai.model"),
    ({"providers__openai__api_key_env": "sk-123"}, "api_key_env"),
    ({"providers__gemini__timeout_seconds": 0}, "timeout_seconds"),
    ({"providers__gemini__concurrency": 1.5}, "concurrency"),
    ({"providers__gemini__max_output_tokens": True}, "max_output_tokens"),
    ({"providers__openai__web_search": {"use_market_location": "yes"}}, "use_market_location"),
    ({"providers__openai__temperature": 0.2}, "unknown keys"),
    ({"retry__max_attempts": -1}, "retry.max_attempts"),
    ({"retry__initial_backoff_seconds": 20}, "must not exceed"),
    ({"pilot__query_ids": ["Q001", "Q002"]}, "3-5 query IDs"),
    ({"pilot__query_ids": ["Q001", "Q002", "Q003", "Q004", "Q005", "Q006"]}, "3-5 query IDs"),
    ({"pilot__query_ids": ["Q001", "Q001", "Q002"]}, "duplicates"),
    ({"pilot__query_ids": ["Q001", "Q002", "q3"]}, "look like Q001"),
])
def test_invalid_values_are_reported(changes, message):
    assert any(message in e for e in errors_for(with_changes(**changes)))


def test_unknown_provider_is_rejected():
    raw = copy.deepcopy(VALID)
    raw["providers"]["mistral"] = raw["providers"]["gemini"]
    assert any("unsupported provider" in e for e in errors_for(raw))


def test_all_errors_are_reported_together():
    raw = with_changes(providers__openai__model="", retry__max_attempts=0, pilot__query_ids=[])
    assert len(errors_for(raw)) >= 3


def test_api_key_is_read_from_environment(monkeypatch):
    provider = parse_config(VALID).providers["openai"]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(CollectorConfigError, match="OPENAI_API_KEY is not set"):
        provider.api_key()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert provider.api_key() == "test-key"
    assert "test-key" not in repr(provider)
