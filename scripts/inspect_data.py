"""Quick technical inspection of raw Olist CSVs.

Reports shape, dtypes, null counts, and a small sample for each file.
Purpose: inform schema design before writing SQL DDL.

Usage:
    python scripts/inspect_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("data/raw")


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    return logging.getLogger("inspect_data")


def inspect_file(path: Path, log: logging.Logger) -> None:
    log.info("\n" + "=" * 80)
    log.info("FILE: %s", path.name)
    log.info("=" * 80)

    df = pd.read_csv(path)
    n_rows, n_cols = df.shape
    log.info("Shape: %s rows x %s columns", f"{n_rows:,}", n_cols)

    log.info("\nColumns and dtypes:")
    for col, dtype in df.dtypes.items():
        n_nulls = df[col].isna().sum()
        pct_nulls = (n_nulls / n_rows) * 100 if n_rows else 0
        n_unique = df[col].nunique()
        log.info(
            "  %-40s %-12s nulls=%-7s (%5.1f%%)  unique=%s",
            col,
            str(dtype),
            f"{n_nulls:,}",
            pct_nulls,
            f"{n_unique:,}",
        )

    log.info("\nSample (first 3 rows):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        log.info(df.head(3).to_string())


def main() -> None:
    log = setup_logging()
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        log.error("No CSV files found in %s. Run download_data.py first.", RAW_DATA_DIR)
        return

    log.info("Found %d CSV files in %s", len(csv_files), RAW_DATA_DIR)
    for path in csv_files:
        inspect_file(path, log)


if __name__ == "__main__":
    main()
