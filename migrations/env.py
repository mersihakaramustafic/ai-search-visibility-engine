"""Alembic environment. Reads the connection string from DATABASE_URL (or .env).

The URL is passed straight to SQLAlchemy rather than through alembic.ini, so
passwords containing '%' are not mangled by configparser and are never written
to a config file.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine, pool

from visibility_db.config import DEFAULT_ENV_VAR, describe_url, get_database_url

# `alembic -x db_env=TEST_DATABASE_URL upgrade head` targets another variable.
env_var = context.get_x_argument(as_dictionary=True).get("db_env", DEFAULT_ENV_VAR)


def run_migrations_offline() -> None:
    """Emit SQL to stdout (`alembic upgrade head --sql`) without connecting."""
    context.configure(
        url="postgresql+psycopg://",
        dialect_name="postgresql",
        literal_binds=True,
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = context.config.attributes.get("connection")
    if connection is not None:  # supplied programmatically, e.g. by tests
        context.configure(connection=connection, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()
        return

    url = get_database_url(env_var)
    if os.environ.get("ALEMBIC_QUIET") != "1":
        print(f"Migrating database at {describe_url(url)} (from ${env_var})")
    engine = create_engine(url, poolclass=pool.NullPool, hide_parameters=True)
    with engine.connect() as conn:
        context.configure(connection=conn, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
