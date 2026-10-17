"""Database connection settings, read from the environment.

The connection string comes from DATABASE_URL (or another variable name passed
in), optionally loaded from a .env file at the repository root. It is never
logged: use describe_url() when a human-readable target is needed.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine, make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_VAR = "DATABASE_URL"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class DatabaseConfigError(Exception):
    """Raised when the database connection string is missing or malformed."""


def load_dotenv_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load .env without overriding variables already set in the shell."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional; plain env vars still work
        return
    if path.is_file():
        load_dotenv(path, override=False)


def normalize_url(raw: str) -> URL:
    """Turn a Supabase/libpq connection string into a SQLAlchemy psycopg 3 URL.

    Supabase shows URLs as postgresql://... ; SQLAlchemy needs the driver
    named explicitly. Remote hosts get sslmode=require unless one is given.
    """
    raw = raw.strip()
    if not raw:
        raise DatabaseConfigError("Database URL is empty.")
    if "[YOUR-PASSWORD]" in raw or "<" in raw:
        raise DatabaseConfigError("Database URL still contains a placeholder; paste the real value.")
    try:
        url = make_url(raw)
    except Exception:  # the message may echo the URL, so don't chain it
        raise DatabaseConfigError("Database URL could not be parsed.") from None
    if url.get_backend_name() != "postgresql":
        raise DatabaseConfigError(f"Expected a PostgreSQL URL, got backend '{url.get_backend_name()}'.")
    url = url.set(drivername="postgresql+psycopg")
    if url.host and url.host not in LOCAL_HOSTS and "sslmode" not in url.query:
        url = url.update_query_dict({"sslmode": "require"})
    return url


def get_database_url(env_var: str = DEFAULT_ENV_VAR) -> URL:
    load_dotenv_file()
    raw = os.environ.get(env_var)
    if not raw:
        raise DatabaseConfigError(
            f"{env_var} is not set. Copy .env.example to .env and fill in your Supabase connection string."
        )
    return normalize_url(raw)


def describe_url(url: URL) -> str:
    """Host, port, and database only: no username or password."""
    return f"{url.host}:{url.port or 5432}/{url.database}"


def make_engine(env_var: str = DEFAULT_ENV_VAR) -> Engine:
    # hide_parameters keeps row values out of error messages and logs.
    return create_engine(get_database_url(env_var), pool_pre_ping=True, hide_parameters=True)
