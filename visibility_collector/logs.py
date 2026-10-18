"""Structured (one JSON object per line) logging for the collector.

Log calls pass fields via extra={"fields": {...}}. Only request metadata is
logged: never API keys, connection strings, or full response bodies.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

LOGGER_NAME = "visibility_collector"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> logging.Logger:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers[:] = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False
    # The SDKs log full request details at DEBUG; keep them quiet.
    for noisy in ("httpx", "httpcore", "openai", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log(level: int, event: str, **fields) -> None:
    get_logger().log(level, event, extra={"fields": fields})
