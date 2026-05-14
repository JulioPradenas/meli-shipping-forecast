"""End-to-end integration test for the full feature engineering pipeline.

Loads real shipment data from the local SQLite database, applies the
default feature pipeline used in modeling, and verifies basic invariants:

* All declared features appear in the output.
* Row count is preserved.
* No accidental NaN explosion outside the expected warm-up region.
* Lag-7 values match the source data (sanity check for leakage).

This test requires the SQLite database at ``data/processed/shipping.db``
to exist and be populated. It is skipped automatically if not present,
so the rest of the test suite can run in environments without the data.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from shipping_forecast.features import (
    CalendarFeatures,
    EventFeatures,
    FeaturePipeline,
    HolidayFeatures,
    LagFeatures,
    RollingFeatures,
    TrendFeatures,
)

DB_PATH = Path("data/processed/shipping.db")


pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="SQLite database not found. Run `python scripts/load_data.py` first.",
)


@pytest.fixture(scope="module")
def shipments_df() -> pd.DataFrame:
    """Load the daily shipments table from the SQLite database."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql(
        """
        SELECT shipment_date, customer_state, n_shipments
        FROM fact_daily_shipments_by_state
        ORDER BY customer_state, shipment_date
        """,
        engine,
        parse_dates=["shipment_date"],
    )
    return df


@pytest.fixture
def default_pipeline() -> FeaturePipeline:
    """The feature pipeline that will be used in modeling.

    Mirrors the modeling decisions documented in EDA_SUMMARY.md:

    * Lags 1, 7, 14, 28 (lag_7 is expected to be dominant)
    * Rolling mean 7, 14, 28 + std 7 (trend + volatility)
    * Calendar features for weekly and monthly seasonality
    * Brazilian operational calendar (sundays, holidays, Carnival)
    * Black Friday and Dia dos Namorados commercial events
    * Long-term trend anchored at 2017-01-01
    """
    return FeaturePipeline(
        [
            LagFeatures(lags=[1, 7, 14, 28]),
            RollingFeatures(windows=[7, 14, 28], stats=["mean"]),
            RollingFeatures(windows=[7], stats=["std"]),
            CalendarFeatures(
                features=["day_of_week", "day_of_month", "month", "quarter", "is_weekend"]
            ),
            HolidayFeatures(),
            EventFeatures(),
            TrendFeatures(reference_date=date(2017, 1, 1)),
        ]
    )


# ----------------------------------------------------------------------- tests


@pytest.mark.integration
def test_pipeline_runs_on_real_data(
    shipments_df: pd.DataFrame, default_pipeline: FeaturePipeline
) -> None:
    """The full pipeline must run successfully on the real dataset."""
    out = default_pipeline.transform(shipments_df)
    # Sanity: not empty
    assert len(out) > 0
    # Row count preserved (no rows dropped)
    assert len(out) == len(shipments_df)


@pytest.mark.integration
def test_all_declared_features_present(
    shipments_df: pd.DataFrame, default_pipeline: FeaturePipeline
) -> None:
    """Every column declared in the pipeline must appear in the output."""
    out = default_pipeline.transform(shipments_df)
    for col in default_pipeline.feature_names:
        assert col in out.columns, f"Missing column from pipeline: {col}"


@pytest.mark.integration
def test_original_columns_preserved(
    shipments_df: pd.DataFrame, default_pipeline: FeaturePipeline
) -> None:
    """The transform must keep the original three columns of the input."""
    out = default_pipeline.transform(shipments_df)
    for col in ["shipment_date", "customer_state", "n_shipments"]:
        assert col in out.columns


@pytest.mark.integration
def test_lag_7_matches_source_data(
    shipments_df: pd.DataFrame, default_pipeline: FeaturePipeline
) -> None:
    """Verify lag_7 by hand for a single state: it must equal n_shipments
    shifted exactly 7 rows back within that state.

    This is the critical leakage check on real data: if lag_7 leaks the
    future, this assertion will fail.
    """
    out = default_pipeline.transform(shipments_df)
    sp = out[out["customer_state"] == "SP"].sort_values("shipment_date").reset_index(drop=True)
    # The 7th row's lag_7 should equal the 0th row's n_shipments
    assert sp["lag_7"].iloc[7] == sp["n_shipments"].iloc[0]
    # The 100th row's lag_7 should equal the 93rd row's n_shipments
    assert sp["lag_7"].iloc[100] == sp["n_shipments"].iloc[93]


@pytest.mark.integration
def test_warmup_nans_are_bounded(
    shipments_df: pd.DataFrame, default_pipeline: FeaturePipeline
) -> None:
    """NaN values from lags/rollings should be limited to the warm-up period.

    For each state, the first 28 days (longest lag) may have NaN in lag_28.
    Beyond day 28, lag_28 should be fully populated.
    """
    out = default_pipeline.transform(shipments_df)

    # Pick one state, check that lag_28 has values from row 28 onward
    sp = out[out["customer_state"] == "SP"].sort_values("shipment_date").reset_index(drop=True)
    # First 28 rows: lag_28 can be NaN
    # From row 28 onward: lag_28 must be defined
    assert sp["lag_28"].iloc[28:].notna().all()


@pytest.mark.integration
def test_holiday_features_match_known_dates(
    shipments_df: pd.DataFrame, default_pipeline: FeaturePipeline
) -> None:
    """Christmas 2017 must be flagged as a holiday in the pipeline output."""
    out = default_pipeline.transform(shipments_df)
    christmas = out[
        (out["shipment_date"] == pd.Timestamp("2017-12-25")) & (out["customer_state"] == "SP")
    ]
    assert len(christmas) == 1
    assert christmas["is_holiday"].iloc[0] == 1
    assert christmas["day_type"].iloc[0] == "holiday"


@pytest.mark.integration
def test_black_friday_features_match_known_dates(
    shipments_df: pd.DataFrame, default_pipeline: FeaturePipeline
) -> None:
    """Black Friday 2017 (Nov 24) must have days_to_black_friday = 0."""
    out = default_pipeline.transform(shipments_df)
    bf = out[(out["shipment_date"] == pd.Timestamp("2017-11-24")) & (out["customer_state"] == "SP")]
    assert len(bf) == 1
    assert bf["days_to_black_friday"].iloc[0] == 0
    assert bf["is_black_friday_window"].iloc[0] == 1


@pytest.mark.integration
def test_pipeline_feature_count(default_pipeline: FeaturePipeline) -> None:
    """Sanity: the default pipeline produces the expected number of features."""
    # 4 lags + (3 means + 1 std) rolling + 5 calendar + 6 holiday + 6 event + 3 trend
    expected = 4 + 4 + 5 + 6 + 6 + 3
    assert len(default_pipeline.feature_names) == expected
