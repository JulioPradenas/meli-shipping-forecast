"""Unit tests for the DataLoader class.

These tests use temporary directories and minimal CSV fixtures to validate
the loader's behaviour without depending on the real Olist dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shipping_forecast.data.loader import DataLoader

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def tmp_raw_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with minimal Olist-like CSVs."""
    raw = tmp_path / "raw"
    raw.mkdir()

    # customers
    pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3"],
            "customer_unique_id": ["u1", "u2", "u3"],
            "customer_zip_code_prefix": [1001, 2002, 3003],
            "customer_city": ["sao paulo", "rio", "belo horizonte"],
            "customer_state": ["SP", "RJ", "MG"],
        }
    ).to_csv(raw / "olist_customers_dataset.csv", index=False)

    # sellers
    pd.DataFrame(
        {
            "seller_id": ["s1", "s2"],
            "seller_zip_code_prefix": [5005, 6006],
            "seller_city": ["campinas", "santos"],
            "seller_state": ["SP", "SP"],
        }
    ).to_csv(raw / "olist_sellers_dataset.csv", index=False)

    # products (with typo'd column names like the real file)
    pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_category_name": ["beleza_saude", "esporte_lazer"],
            "product_name_lenght": [40.0, 46.0],
            "product_description_lenght": [200.0, 250.0],
            "product_photos_qty": [1.0, 1.0],
            "product_weight_g": [225.0, 154.0],
            "product_length_cm": [16.0, 18.0],
            "product_height_cm": [10.0, 9.0],
            "product_width_cm": [14.0, 15.0],
        }
    ).to_csv(raw / "olist_products_dataset.csv", index=False)

    # category translation
    pd.DataFrame(
        {
            "product_category_name": ["beleza_saude", "esporte_lazer"],
            "product_category_name_english": ["health_beauty", "sports_leisure"],
        }
    ).to_csv(raw / "product_category_name_translation.csv", index=False)

    # orders
    pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c2", "c3"],
            "order_status": ["delivered", "shipped", "canceled"],
            "order_purchase_timestamp": [
                "2017-01-01 10:00:00",
                "2017-01-02 11:00:00",
                "2017-01-03 12:00:00",
            ],
            "order_approved_at": [
                "2017-01-01 11:00:00",
                "2017-01-02 12:00:00",
                None,
            ],
            "order_delivered_carrier_date": [
                "2017-01-02 09:00:00",
                "2017-01-03 09:00:00",
                None,
            ],
            "order_delivered_customer_date": [
                "2017-01-05 14:00:00",
                None,
                None,
            ],
            "order_estimated_delivery_date": [
                "2017-01-10 00:00:00",
                "2017-01-11 00:00:00",
                "2017-01-12 00:00:00",
            ],
        }
    ).to_csv(raw / "olist_orders_dataset.csv", index=False)

    # order items
    pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "order_item_id": [1, 2, 1],
            "product_id": ["p1", "p2", "p1"],
            "seller_id": ["s1", "s2", "s1"],
            "shipping_limit_date": [
                "2017-01-08 09:00:00",
                "2017-01-08 09:00:00",
                "2017-01-09 09:00:00",
            ],
            "price": [100.0, 50.0, 75.0],
            "freight_value": [10.0, 5.0, 8.0],
        }
    ).to_csv(raw / "olist_order_items_dataset.csv", index=False)

    return raw


@pytest.fixture
def loader(tmp_raw_dir: Path, tmp_path: Path) -> DataLoader:
    """Create a DataLoader pointing at temporary paths."""
    return DataLoader(
        raw_dir=tmp_raw_dir,
        db_path=tmp_path / "test.db",
        schema_sql=Path("sql/01_create_schema.sql"),
    )


# --------------------------------------------------------------------- tests


def test_create_schema_creates_tables(loader: DataLoader) -> None:
    """create_schema() should create all expected tables."""
    loader.create_schema()
    for table in [
        "dim_customers",
        "dim_sellers",
        "dim_products",
        "fact_orders",
        "fact_order_items",
        "fact_daily_shipments_by_state",
    ]:
        assert loader.count_rows(table) == 0, f"{table} should be empty"


def test_create_schema_raises_if_file_missing(tmp_path: Path) -> None:
    """create_schema() must raise when the schema SQL file does not exist."""
    bad_loader = DataLoader(
        raw_dir=tmp_path,
        db_path=tmp_path / "test.db",
        schema_sql=Path("does/not/exist.sql"),
    )
    with pytest.raises(FileNotFoundError):
        bad_loader.create_schema()


def test_load_customers(loader: DataLoader) -> None:
    """load_customers() returns the correct row count and populates the table."""
    loader.create_schema()
    n = loader.load_customers()
    assert n == 3
    assert loader.count_rows("dim_customers") == 3


def test_load_sellers(loader: DataLoader) -> None:
    """load_sellers() returns the correct row count and populates the table."""
    loader.create_schema()
    n = loader.load_sellers()
    assert n == 2
    assert loader.count_rows("dim_sellers") == 2


def test_load_products_renames_typo_columns(loader: DataLoader) -> None:
    """load_products() must rename product_name_lenght -> product_name_length."""
    loader.create_schema()
    n = loader.load_products()
    assert n == 2
    assert loader.count_rows("dim_products") == 2


def test_load_orders(loader: DataLoader) -> None:
    """load_orders() inserts all rows including those with null dates."""
    loader.create_schema()
    loader.load_customers()  # FK dependency
    n = loader.load_orders()
    assert n == 3


def test_load_order_items(loader: DataLoader) -> None:
    """load_order_items() inserts all rows respecting the composite PK."""
    loader.create_schema()
    loader.load_customers()
    loader.load_sellers()
    loader.load_products()
    loader.load_orders()
    n = loader.load_order_items()
    assert n == 3


def test_load_all_returns_all_counts(loader: DataLoader) -> None:
    """load_all() returns a dict with row counts for every loaded table."""
    loader.create_schema()
    counts = loader.load_all()
    expected_tables = {
        "dim_customers",
        "dim_sellers",
        "dim_products",
        "fact_orders",
        "fact_order_items",
    }
    assert set(counts.keys()) == expected_tables
    assert all(v > 0 for v in counts.values())


def test_build_daily_aggregates_excludes_non_shipments(loader: DataLoader) -> None:
    """The aggregate must exclude canceled orders and null carrier dates.

    With 3 orders in fixtures:
      - o1: delivered, carrier=2017-01-02 -> counts (SP)
      - o2: shipped,   carrier=2017-01-03 -> counts (RJ)
      - o3: canceled,  carrier=NULL       -> excluded

    Expected: total_shipments = 2.
    """
    loader.create_schema()
    loader.load_all()
    n = loader.build_daily_aggregates()
    assert n > 0

    # Verify the total count of shipments equals 2 (o1 + o2)
    with loader._engine.connect() as conn:
        from sqlalchemy import text

        result = conn.execute(text("SELECT SUM(n_shipments) FROM fact_daily_shipments_by_state"))
        total = result.scalar()
        assert total == 2, f"Expected 2 shipments (o1 + o2), got {total}"
