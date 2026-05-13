"""Data loading module for the Olist Brazilian E-Commerce dataset.

This module provides the :class:`DataLoader` class, which orchestrates the
ingestion of raw CSV files into an analytical SQLite database following the
schema defined in ``sql/01_create_schema.sql``.

Typical usage::

    from pathlib import Path
    from shipping_forecast.data.loader import DataLoader

    loader = DataLoader(
        raw_dir=Path("data/raw"),
        db_path=Path("data/processed/shipping.db"),
    )
    loader.create_schema()
    counts = loader.load_all()
    print(counts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, text

logger = logging.getLogger(__name__)


@dataclass
class DataLoader:
    """Loads Olist raw CSVs into the analytical SQLite database.

    Attributes:
        raw_dir: Directory containing the raw CSV files.
        db_path: Path to the target SQLite database file.
        schema_sql: Path to the SQL file containing the schema DDL.
    """

    raw_dir: Path
    db_path: Path
    schema_sql: Path = field(default=Path("sql/01_create_schema.sql"))
    daily_aggregates_sql: Path = field(default=Path("sql/02_build_daily_shipments.sql"))
    features_sql: Path = field(default=Path("sql/03_lag_features.sql"))

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: Engine = create_engine(f"sqlite:///{self.db_path}")

    # ------------------------------------------------------------------ schema

    def create_schema(self) -> None:
        """Execute the schema DDL script.

        Raises:
            FileNotFoundError: If ``schema_sql`` does not exist.
        """
        if not self.schema_sql.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_sql}")

        ddl = self.schema_sql.read_text(encoding="utf-8")
        logger.info("Executing schema DDL from %s", self.schema_sql)

        with self._engine.begin() as conn:
            # SQLAlchemy's text() can't execute multi-statement scripts directly
            # on SQLite, so we use the raw DBAPI connection for executescript.
            raw_conn = conn.connection
            raw_conn.executescript(ddl)  # type: ignore[attr-defined]

        logger.info("Schema created at %s", self.db_path)

    # ------------------------------------------------------------- dimensions

    def load_customers(self) -> int:
        """Load customers CSV into ``dim_customers``. Returns rows inserted."""
        df = pd.read_csv(self.raw_dir / "olist_customers_dataset.csv")
        return self._write(df, "dim_customers")

    def load_sellers(self) -> int:
        """Load sellers CSV into ``dim_sellers``. Returns rows inserted."""
        df = pd.read_csv(self.raw_dir / "olist_sellers_dataset.csv")
        return self._write(df, "dim_sellers")

    def load_products(self) -> int:
        """Load products with English category names into ``dim_products``.

        Joins the raw products CSV with the category-name translation CSV
        and renames the typo'd column ``product_name_lenght`` to
        ``product_name_length``.
        """
        products = pd.read_csv(self.raw_dir / "olist_products_dataset.csv")
        translation = pd.read_csv(self.raw_dir / "product_category_name_translation.csv")

        # Left join so products without translation are kept (NULL english name)
        merged = products.merge(translation, on="product_category_name", how="left")
        merged = merged.rename(
            columns={
                "product_name_lenght": "product_name_length",
                "product_description_lenght": "product_description_length",
                "product_category_name_english": "product_category_name_en",
            }
        )
        return self._write(merged, "dim_products")

    # ------------------------------------------------------------------ facts

    def load_orders(self) -> int:
        """Load orders CSV into ``fact_orders``. Returns rows inserted."""
        df = pd.read_csv(self.raw_dir / "olist_orders_dataset.csv")
        return self._write(df, "fact_orders")

    def load_order_items(self) -> int:
        """Load order items CSV into ``fact_order_items``. Returns rows inserted."""
        df = pd.read_csv(self.raw_dir / "olist_order_items_dataset.csv")
        return self._write(df, "fact_order_items")

    # ------------------------------------------------------------ orchestration

    def load_all(self) -> dict[str, int]:
        """Load all tables in dependency order. Returns row counts per table.

        Dimensions must be loaded before facts to honour foreign keys.
        """
        logger.info("Starting full load")
        counts = {
            "dim_customers": self.load_customers(),
            "dim_sellers": self.load_sellers(),
            "dim_products": self.load_products(),
            "fact_orders": self.load_orders(),
            "fact_order_items": self.load_order_items(),
        }
        for table, n in counts.items():
            logger.info("  %s rows -> %s", f"{n:>7,}", table)
        return counts

    # ----------------------------------------------------------------- helpers

    def _write(self, df: pd.DataFrame, table: str) -> int:
        """Write a DataFrame to the given table, replacing all existing rows."""
        n = len(df)
        logger.info("Writing %s rows to %s", f"{n:,}", table)
        df.to_sql(table, self._engine, if_exists="append", index=False)
        return n

    def count_rows(self, table: str) -> int:
        """Return the number of rows currently in ``table``."""
        with self._engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            return int(result.scalar() or 0)

    def build_daily_aggregates(self) -> int:
        """Execute the daily aggregates SQL script.

        Populates ``fact_daily_shipments_by_state`` from ``fact_orders`` and
        ``dim_customers``. Returns the number of rows in the resulting table.

        Raises:
            FileNotFoundError: If ``daily_aggregates_sql`` does not exist.
        """
        if not self.daily_aggregates_sql.exists():
            raise FileNotFoundError(f"Aggregates SQL file not found: {self.daily_aggregates_sql}")

        sql = self.daily_aggregates_sql.read_text(encoding="utf-8")
        logger.info("Building daily aggregates from %s", self.daily_aggregates_sql)

        with self._engine.begin() as conn:
            raw_conn = conn.connection
            raw_conn.executescript(sql)  # type: ignore[attr-defined]

        n = self.count_rows("fact_daily_shipments_by_state")
        logger.info("fact_daily_shipments_by_state populated with %s rows", f"{n:,}")
        return n

    def build_features(self) -> int:
        """Execute the lag/rolling features SQL script.

        Populates ``fact_daily_shipments_features`` from
        ``fact_daily_shipments_by_state`` using SQL window functions for
        lag and rolling stats. Returns the number of rows in the resulting
        table.

        Raises:
            FileNotFoundError: If ``features_sql`` does not exist.
        """
        if not self.features_sql.exists():
            raise FileNotFoundError(f"Features SQL file not found: {self.features_sql}")

        sql = self.features_sql.read_text(encoding="utf-8")
        logger.info("Building lag/rolling features from %s", self.features_sql)

        with self._engine.begin() as conn:
            raw_conn = conn.connection
            raw_conn.executescript(sql)  # type: ignore[attr-defined]

        n = self.count_rows("fact_daily_shipments_features")
        logger.info("fact_daily_shipments_features populated with %s rows", f"{n:,}")
        return n
