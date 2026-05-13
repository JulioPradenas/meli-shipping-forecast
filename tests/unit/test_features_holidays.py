"""Unit tests for HolidayFeatures and the underlying calendar utilities."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from shipping_forecast.features.holidays import HolidayFeatures
from shipping_forecast.utils.calendar_br import (
    build_br_operational_holidays,
    compute_easter,
)

# --------------------------------------------------------------------- easter


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        # Known Easter dates verified against multiple sources
        (2017, date(2017, 4, 16)),
        (2018, date(2018, 4, 1)),
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
    ],
)
def test_compute_easter_known_years(year: int, expected: date) -> None:
    """The Anonymous Gregorian algorithm must match known Easter dates."""
    assert compute_easter(year) == expected


# ------------------------------------------------------- operational calendar


def test_carnival_2017_is_included() -> None:
    """Carnival 2017 was Feb 27-28. Both must be in the operational holiday set."""
    op_holidays = build_br_operational_holidays([2017])
    assert date(2017, 2, 27) in op_holidays  # Carnival Monday
    assert date(2017, 2, 28) in op_holidays  # Carnival Tuesday


def test_carnival_2018_is_included() -> None:
    """Carnival 2018 was Feb 12-13."""
    op_holidays = build_br_operational_holidays([2018])
    assert date(2018, 2, 12) in op_holidays
    assert date(2018, 2, 13) in op_holidays


def test_corpus_christi_is_included() -> None:
    """Corpus Christi 2017 was June 15; 2018 was May 31."""
    op_holidays = build_br_operational_holidays([2017, 2018])
    assert date(2017, 6, 15) in op_holidays
    assert date(2018, 5, 31) in op_holidays


def test_federal_holidays_are_included() -> None:
    """Standard federal holidays must come from the holidays library."""
    op_holidays = build_br_operational_holidays([2017])
    assert date(2017, 1, 1) in op_holidays  # New Year's Day
    assert date(2017, 5, 1) in op_holidays  # Labor Day
    assert date(2017, 9, 7) in op_holidays  # Independence Day
    assert date(2017, 12, 25) in op_holidays  # Christmas


def test_regular_day_is_not_in_holidays() -> None:
    """A random Wednesday should not be in the operational holiday set."""
    op_holidays = build_br_operational_holidays([2017])
    assert date(2017, 7, 5) not in op_holidays  # arbitrary Wednesday


# ----------------------------------------------------------- holiday features


@pytest.fixture
def calendar_check_df() -> pd.DataFrame:
    """A small DataFrame with hand-picked dates covering all day types.

    Dates chosen:
      2017-11-24: Black Friday (regular Friday, not a holiday)
      2017-11-25: Saturday after Black Friday
      2017-11-26: Sunday after Black Friday
      2017-12-25: Christmas (Monday holiday)
      2018-02-12: Carnival Monday
      2018-04-02: Regular Monday
    """
    return pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(
                [
                    "2017-11-24",
                    "2017-11-25",
                    "2017-11-26",
                    "2017-12-25",
                    "2018-02-12",
                    "2018-04-02",
                ]
            ),
            "n_shipments": [325, 70, 0, 0, 0, 200],
        }
    )


def test_all_feature_columns_added(calendar_check_df: pd.DataFrame) -> None:
    """All 6 declared feature columns must be present after transform."""
    out = HolidayFeatures().transform(calendar_check_df)
    for col in [
        "is_holiday",
        "is_business_day",
        "is_saturday",
        "is_sunday",
        "is_operational",
        "day_type",
    ]:
        assert col in out.columns


def test_day_type_classification(calendar_check_df: pd.DataFrame) -> None:
    """day_type must label each date correctly."""
    out = HolidayFeatures().transform(calendar_check_df)
    expected = [
        "business_day",  # 2017-11-24 Friday, no holiday
        "saturday",  # 2017-11-25 Saturday
        "sunday",  # 2017-11-26 Sunday
        "holiday",  # 2017-12-25 Christmas
        "holiday",  # 2018-02-12 Carnival Monday
        "business_day",  # 2018-04-02 regular Monday
    ]
    assert out["day_type"].astype(str).tolist() == expected


def test_boolean_features_are_consistent(calendar_check_df: pd.DataFrame) -> None:
    """For each row, exactly one of business_day/saturday/sunday/holiday is 1.

    is_operational should equal business_day OR saturday.
    """
    out = HolidayFeatures().transform(calendar_check_df)

    # Mutual exclusivity: rows sum to 1 across the four bool flags
    flags = out[["is_business_day", "is_saturday", "is_sunday", "is_holiday"]]
    assert (flags.sum(axis=1) == 1).all()

    # is_operational = business_day OR saturday
    expected_operational = out["is_business_day"] | out["is_saturday"]
    assert (out["is_operational"] == expected_operational).all()


def test_holiday_overrides_weekday(calendar_check_df: pd.DataFrame) -> None:
    """A Monday that is Christmas must be classified as holiday, not business_day."""
    out = HolidayFeatures().transform(calendar_check_df)
    christmas_row = out[out["shipment_date"] == pd.Timestamp("2017-12-25")].iloc[0]
    assert christmas_row["day_type"] == "holiday"
    assert christmas_row["is_holiday"] == 1
    assert christmas_row["is_business_day"] == 0


def test_carnival_classified_as_holiday(calendar_check_df: pd.DataFrame) -> None:
    """Carnival Monday must be 'holiday', not 'business_day'."""
    out = HolidayFeatures().transform(calendar_check_df)
    carnival_row = out[out["shipment_date"] == pd.Timestamp("2018-02-12")].iloc[0]
    assert carnival_row["day_type"] == "holiday"
    assert carnival_row["is_holiday"] == 1


def test_input_dataframe_is_not_mutated(calendar_check_df: pd.DataFrame) -> None:
    """The transform must never modify the input DataFrame in place."""
    original_cols = calendar_check_df.columns.tolist()
    HolidayFeatures().transform(calendar_check_df)
    assert calendar_check_df.columns.tolist() == original_cols


def test_feature_names_property() -> None:
    """feature_names returns the 6 declared features in order."""
    builder = HolidayFeatures()
    assert builder.feature_names == [
        "is_holiday",
        "is_business_day",
        "is_saturday",
        "is_sunday",
        "is_operational",
        "day_type",
    ]
