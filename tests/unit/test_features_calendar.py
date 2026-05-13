"""Unit tests for CalendarFeatures."""

from __future__ import annotations

import pandas as pd
import pytest

from shipping_forecast.features.calendar import CalendarFeatures

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def known_dates_df() -> pd.DataFrame:
    """Hand-picked dates where we know the expected calendar values.

    Picked so each date hits a different combination of features:
      2024-01-01: Monday (dow=0), 1st of month, Q1, week 1
      2024-01-31: Wednesday (dow=2), end of January, Q1
      2024-07-13: Saturday (dow=5), Q3, weekend
      2024-12-31: Tuesday (dow=1), end of year, Q4
    """
    return pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(
                ["2024-01-01", "2024-01-31", "2024-07-13", "2024-12-31"]
            ),
            "n_shipments": [100, 200, 50, 300],
        }
    )


# --------------------------------------------------------------------- tests


def test_default_features_added(known_dates_df: pd.DataFrame) -> None:
    """With defaults, the standard 5 features should be added."""
    builder = CalendarFeatures()
    out = builder.transform(known_dates_df)

    for col in ["day_of_week", "day_of_month", "month", "quarter", "is_weekend"]:
        assert col in out.columns


def test_day_of_week_values(known_dates_df: pd.DataFrame) -> None:
    """Mon=0 ... Sun=6. Verify against known dates."""
    builder = CalendarFeatures(features=["day_of_week"])
    out = builder.transform(known_dates_df)

    # 2024-01-01 = Monday (0)
    # 2024-01-31 = Wednesday (2)
    # 2024-07-13 = Saturday (5)
    # 2024-12-31 = Tuesday (1)
    assert out["day_of_week"].tolist() == [0, 2, 5, 1]


def test_month_and_quarter(known_dates_df: pd.DataFrame) -> None:
    """Month is 1-12, quarter is 1-4."""
    builder = CalendarFeatures(features=["month", "quarter"])
    out = builder.transform(known_dates_df)

    assert out["month"].tolist() == [1, 1, 7, 12]
    assert out["quarter"].tolist() == [1, 1, 3, 4]


def test_is_weekend_binary(known_dates_df: pd.DataFrame) -> None:
    """is_weekend = 1 only for Saturday and Sunday."""
    builder = CalendarFeatures(features=["is_weekend"])
    out = builder.transform(known_dates_df)

    # Mon, Wed, Sat, Tue
    assert out["is_weekend"].tolist() == [0, 0, 1, 0]


def test_is_month_start_and_end() -> None:
    """is_month_start/end correctly identify boundary days."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(
                ["2024-02-01", "2024-02-15", "2024-02-29"]
            ),  # leap year
            "n_shipments": [1, 2, 3],
        }
    )
    builder = CalendarFeatures(features=["is_month_start", "is_month_end"])
    out = builder.transform(df)

    assert out["is_month_start"].tolist() == [1, 0, 0]
    assert out["is_month_end"].tolist() == [0, 0, 1]


def test_dtypes_are_compact() -> None:
    """Calendar features should use small integer dtypes for memory efficiency."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(["2024-01-01"]),
            "n_shipments": [1],
        }
    )
    builder = CalendarFeatures(features=["day_of_week", "month", "year", "is_weekend"])
    out = builder.transform(df)

    # int8 ranges for things bounded by small numbers
    assert out["day_of_week"].dtype == "int8"
    assert out["month"].dtype == "int8"
    assert out["is_weekend"].dtype == "int8"
    # int16 for year (covers 1900-2099 easily)
    assert out["year"].dtype == "int16"


def test_string_dates_are_handled() -> None:
    """If sort_col is a string column, transform should still work."""
    df = pd.DataFrame(
        {
            "shipment_date": ["2024-01-01", "2024-07-04"],  # strings, not datetime
            "n_shipments": [10, 20],
        }
    )
    builder = CalendarFeatures(features=["day_of_week"])
    out = builder.transform(df)

    # 2024-01-01=Monday(0), 2024-07-04=Thursday(3)
    assert out["day_of_week"].tolist() == [0, 3]


def test_validation_rejects_empty_features() -> None:
    """An empty features list must raise ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        CalendarFeatures(features=[])


def test_validation_rejects_unsupported_feature() -> None:
    """Features outside the supported set must raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported"):
        CalendarFeatures(features=["day_of_week", "fortnight"])


def test_input_dataframe_is_not_mutated(known_dates_df: pd.DataFrame) -> None:
    """The transform must never modify the input DataFrame in place."""
    original_cols = known_dates_df.columns.tolist()

    CalendarFeatures().transform(known_dates_df)

    assert known_dates_df.columns.tolist() == original_cols


def test_only_requested_features_added(known_dates_df: pd.DataFrame) -> None:
    """If you ask for 1 feature, only 1 column should be added."""
    builder = CalendarFeatures(features=["month"])
    out = builder.transform(known_dates_df)

    # Should have original columns + just 'month'
    new_cols = set(out.columns) - set(known_dates_df.columns)
    assert new_cols == {"month"}


def test_feature_names_property() -> None:
    """feature_names returns exactly what was configured."""
    builder = CalendarFeatures(features=["day_of_week", "month"])
    assert builder.feature_names == ["day_of_week", "month"]
