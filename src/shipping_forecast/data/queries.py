"""Read queries against the analytical SQLite database.

This module provides standalone functions for reading data from the
shipping forecasting database. It complements :mod:`shipping_forecast.data.loader`,
which handles ingestion (write) workflows.

The functions here are stateless and reusable across pipelines, scripts,
and notebooks. Keeping reads separate from writes makes it easier to
test query logic without spinning up an ingestion pipeline.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def load_panel(db_path: Path) -> pd.DataFrame:
    """Load the daily shipments panel from SQLite.

    Args:
        db_path: Path to the analytical SQLite database. Must contain
            the table ``fact_daily_shipments_by_state``.

    Returns:
        DataFrame with columns ``[shipment_date, customer_state, n_shipments]``,
        sorted by ``(customer_state, shipment_date)``. The ``shipment_date``
        column is parsed as datetime.

    Raises:
        FileNotFoundError: If ``db_path`` does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Shipping DB not found at {db_path}. Run scripts/load_data.py first."
        )
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT shipment_date, customer_state, n_shipments "
            "FROM fact_daily_shipments_by_state "
            "ORDER BY customer_state, shipment_date",
            conn,
            parse_dates=["shipment_date"],
        )
    return df
