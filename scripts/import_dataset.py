#!/usr/bin/env python3
"""Load the generated CSVs from data/processed/ into the database.

Tables are loaded in foreign-key order inside one transaction, so a failure
leaves the database unchanged:

    brands.csv                     -> brands
    queries.csv                    -> queries
    collection_manifest.csv        -> collection_requests
    recommendation_candidates.csv  -> recommendation_candidates

Rows are upserted by primary key, so re-running is safe and picks up edits.
Collection requests keep their database status on re-import: progress made by
a collector is never reset to the CSV's 'pending'. Rows that exist in the
database but not in the CSV are reported, never deleted.

Usage:
    python scripts/import_dataset.py [--data-dir PATH] [--db-env VAR] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import bindparam, text  # noqa: E402
from sqlalchemy.dialects.postgresql import JSONB  # noqa: E402

from visibility_db.config import DEFAULT_ENV_VAR, DatabaseConfigError, describe_url, make_engine  # noqa: E402

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class TableSpec:
    csv_name: str
    table: str
    key: str
    columns: tuple[str, ...]
    bool_columns: tuple[str, ...] = ()
    int_columns: tuple[str, ...] = ()
    json_columns: tuple[str, ...] = ()
    # Columns written on insert but left alone when the row already exists.
    insert_only: tuple[str, ...] = ()


SPECS = [
    TableSpec(
        csv_name="brands.csv",
        table="brands",
        key="brand_id",
        columns=("brand_id", "brand_name", "canonical_domain", "aliases", "description",
                 "product_category", "is_primary_brand"),
        bool_columns=("is_primary_brand",),
        json_columns=("aliases",),
    ),
    TableSpec(
        csv_name="queries.csv",
        table="queries",
        key="query_id",
        columns=("query_id", "query_text", "intent", "topic", "target_audience", "language", "market",
                 "contains_brand_name", "named_brands", "query_family_id"),
        bool_columns=("contains_brand_name",),
        json_columns=("named_brands",),
    ),
    TableSpec(
        csv_name="collection_manifest.csv",
        table="collection_requests",
        key="request_id",
        columns=("request_id", "query_id", "provider", "run_number", "market", "language",
                 "search_enabled", "status"),
        bool_columns=("search_enabled",),
        int_columns=("run_number",),
        insert_only=("status",),
    ),
    TableSpec(
        csv_name="recommendation_candidates.csv",
        table="recommendation_candidates",
        key="candidate_id",
        columns=("candidate_id", "target_brand", "action_type", "topic", "query_family_id",
                 "evidence_pattern", "suggested_action", "evidence_required", "estimated_effort",
                 "hypothesis", "is_synthetic"),
        bool_columns=("is_synthetic",),
    ),
]


class DatasetImportError(Exception):
    """Raised when a CSV is missing or has unexpected content."""


def parse_bool(value: str, column: str, row_number: int) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise DatasetImportError(f"row {row_number}: {column} must be 'true' or 'false', got {value!r}")


def read_rows(path: Path, spec: TableSpec) -> list[dict]:
    if not path.is_file():
        raise DatasetImportError(f"{path} not found; run scripts/generate_initial_dataset.py first")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(spec.columns) - set(reader.fieldnames or ())
        if missing:
            raise DatasetImportError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        rows = []
        for row_number, raw in enumerate(reader, start=2):
            row = {c: raw[c] for c in spec.columns}
            for c in spec.bool_columns:
                row[c] = parse_bool(row[c], c, row_number)
            for c in spec.int_columns:
                row[c] = int(row[c])
            for c in spec.json_columns:
                row[c] = json.loads(row[c])
            rows.append(row)
    keys = [r[spec.key] for r in rows]
    if len(keys) != len(set(keys)):
        raise DatasetImportError(f"{path.name} has duplicate {spec.key} values")
    return rows


def upsert_statement(spec: TableSpec):
    updated = [c for c in spec.columns if c != spec.key and c not in spec.insert_only]
    # Skip the write when nothing changed, so updated_at only moves on real edits.
    changed = " OR ".join(f"t.{c} IS DISTINCT FROM EXCLUDED.{c}" for c in updated)
    sql = (
        f"INSERT INTO {spec.table} AS t ({', '.join(spec.columns)}) "
        f"VALUES ({', '.join(':' + c for c in spec.columns)}) "
        f"ON CONFLICT ({spec.key}) DO UPDATE SET "
        f"{', '.join(f'{c} = EXCLUDED.{c}' for c in updated)} "
        f"WHERE {changed}"
    )
    return text(sql).bindparams(*(bindparam(c, type_=JSONB) for c in spec.json_columns))


def import_all(data_dir: Path, db_env: str, dry_run: bool) -> None:
    loaded = [(spec, read_rows(data_dir / spec.csv_name, spec)) for spec in SPECS]
    engine = make_engine(db_env)
    print(f"{'Checking' if dry_run else 'Importing into'} {describe_url(engine.url)} (from ${db_env})")
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            for spec, rows in loaded:
                before = set(conn.execute(text(f"SELECT {spec.key} FROM {spec.table}")).scalars())
                csv_keys = {r[spec.key] for r in rows}
                conn.execute(upsert_statement(spec), rows)
                new = len(csv_keys - before)
                extra = len(before - csv_keys)
                line = f"  {spec.table:27} {len(rows):4} rows from CSV, {new:4} new"
                if extra:
                    line += f", {extra} in DB but not in CSV (left as is)"
                print(line)
        except Exception:
            trans.rollback()
            raise
        if dry_run:
            trans.rollback()
            print("Dry run: rolled back, nothing written.")
        else:
            trans.commit()
            print("Done.")
    engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with the generated CSVs")
    parser.add_argument("--db-env", default=DEFAULT_ENV_VAR, help="Environment variable holding the database URL")
    parser.add_argument("--dry-run", action="store_true", help="Run the import, report counts, then roll back")
    args = parser.parse_args(argv)
    try:
        import_all(args.data_dir, args.db_env, args.dry_run)
    except (DatasetImportError, DatabaseConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
