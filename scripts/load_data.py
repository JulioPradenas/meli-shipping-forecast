"""Load Olist raw CSVs into the analytical SQLite database.

This script orchestrates the DataLoader to:
  1. (Re)create the schema from sql/01_create_schema.sql
  2. Load all dimensions and facts in dependency order
  3. Verify final row counts

Usage:
    python scripts/load_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from shipping_forecast.data.loader import DataLoader

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/processed/shipping.db")
SCHEMA_SQL = Path("sql/01_create_schema.sql")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logging()
    log = logging.getLogger("load_data")

    log.info("Initializing DataLoader")
    loader = DataLoader(raw_dir=RAW_DIR, db_path=DB_PATH, schema_sql=SCHEMA_SQL)

    log.info("Creating schema")
    loader.create_schema()

    log.info("Loading all tables")
    counts = loader.load_all()
    log.info("Building daily aggregates")
    n_agg = loader.build_daily_aggregates()
    counts["fact_daily_shipments_by_state"] = n_agg
    log.info("Building lag/rolling features")
    n_feat = loader.build_features()
    counts["fact_daily_shipments_features"] = n_feat
    log.info("Final row counts:")
    for table, n in counts.items():
        actual = loader.count_rows(table)
        match = "ok" if actual == n else "MISMATCH"
        log.info(
            "  %-20s expected=%-10s actual=%-10s [%s]",
            table,
            f"{n:,}",
            f"{actual:,}",
            match,
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
