"""Unit tests for TrendFeatures."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from shipping_forecast.features.trend import TrendFeatures

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def known_dates_df() -> pd.DataFrame:
    """A small DataFrame with dates at known offsets from 2017-01-01."""
    return pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(
                [
                    "2017-01-01",  # day 0, month 0, jan 1st
                    "2017-01-31",  # day 30, month 0, late jan
                    "2017-02-01",  # day 31, month 1, feb 1st
                    "2018-01-01",  # day 365, month 12, full year later
                ]
            ),
            "n_shipments": [10, 20, 30, 40],
        }
    )


# --------------------------------------------------------------------- tests


def test_days_since_start_with_default_reference(known_dates_df: pd.DataFrame) -> None:
    """days_since_start counts days from 2017-01-01 (default reference)."""
    builder = TrendFeatures(features=["days_since_start"])
    out = builder.transform(known_dates_df)
    # 2017-01-01 -> 0
    # 2017-01-31 -> 30
    # 2017-02-01 -> 31
    # 2018-01-01 -> 365
    assert out["days_since_start"].tolist() == [0, 30, 31, 365]


def test_days_since_start_with_custom_reference() -> None:
    """A custom reference_date shifts the day counter accordingly."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(["2020-01-01", "2020-01-11"]),
            "n_shipments": [1, 2],
        }
    )
    builder = TrendFeatures(
        features=["days_since_start"],
        reference_date=date(2020, 1, 1),
    )
    out = builder.transform(df)
    assert out["days_since_start"].tolist() == [0, 10]


def test_month_index(known_dates_df: pd.DataFrame) -> None:
    """month_index increments by 1 each calendar month."""
    builder = TrendFeatures(features=["month_index"])
    out = builder.transform(known_dates_df)
    # 2017-01 -> 0, 2017-01 -> 0, 2017-02 -> 1, 2018-01 -> 12
    assert out["month_index"].tolist() == [0, 0, 1, 12]


def test_year_progress_bounds() -> None:
    """year_progress should be in [0, 1) for every date in the year."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(["2020-01-01", "2020-07-01", "2020-12-31"]),
            "n_shipments": [1, 2, 3],
        }
    )
    out = TrendFeatures(features=["year_progress"]).transform(df)
    progress = out["year_progress"].tolist()
    # Jan 1 -> 0.0
    assert progress[0] == pytest.approx(0.0)
    # Mid-year somewhere around 0.5
    assert 0.4 < progress[1] < 0.6
    # Dec 31 close to but below 1.0 (it's 365/365 in leap year safe formula)
    assert progress[2] >= 0.9


def test_all_features_added_by_default(known_dates_df: pd.DataFrame) -> None:
    """With default args, all three features must be present."""
    out = TrendFeatures().transform(known_dates_df)
    for col in ["days_since_start", "month_index", "year_progress"]:
        assert col in out.columns


def test_only_requested_features_added() -> None:
    """Requesting a subset must add only those columns."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(["2017-01-01"]),
            "n_shipments": [10],
        }
    )
    out = TrendFeatures(features=["month_index"]).transform(df)
    assert "month_index" in out.columns
    assert "days_since_start" not in out.columns
    assert "year_progress" not in out.columns


def test_dtypes_are_compact() -> None:
    """Trend features should use compact dtypes for memory efficiency."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(["2017-01-01"]),
            "n_shipments": [10],
        }
    )
    out = TrendFeatures().transform(df)
    assert out["days_since_start"].dtype == "int32"
    assert out["month_index"].dtype == "int16"
    assert out["year_progress"].dtype == "float32"


def test_validation_rejects_empty_features() -> None:
    """An empty features list must raise ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        TrendFeatures(features=[])


def test_validation_rejects_unsupported_feature() -> None:
    """Unknown feature names must raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported"):
        TrendFeatures(features=["days_since_start", "moon_phase"])


def test_input_dataframe_is_not_mutated(known_dates_df: pd.DataFrame) -> None:
    """The transform must never modify the input DataFrame in place."""
    original_cols = known_dates_df.columns.tolist()
    TrendFeatures().transform(known_dates_df)
    assert known_dates_df.columns.tolist() == original_cols


def test_feature_names_property() -> None:
    """feature_names should return the configured features."""
    builder = TrendFeatures(features=["days_since_start", "month_index"])
    assert builder.feature_names == ["days_since_start", "month_index"]
