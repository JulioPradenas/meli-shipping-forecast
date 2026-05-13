"""Unit tests for EventFeatures and event-date utilities."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from shipping_forecast.features.events import (
    EventFeatures,
    _signed_distance_to_nearest,
)
from shipping_forecast.utils.calendar_br import (
    get_black_friday,
    get_dia_dos_namorados,
)

# --------------------------------------------------- event date utilities


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2017, date(2017, 11, 24)),
        (2018, date(2018, 11, 23)),
        (2019, date(2019, 11, 29)),
        (2024, date(2024, 11, 29)),
    ],
)
def test_get_black_friday_known_years(year: int, expected: date) -> None:
    """Black Friday is the last Friday of November."""
    assert get_black_friday(year) == expected


def test_get_dia_dos_namorados_is_always_june_12() -> None:
    """Dia dos Namorados is fixed on June 12 every year."""
    assert get_dia_dos_namorados(2017) == date(2017, 6, 12)
    assert get_dia_dos_namorados(2018) == date(2018, 6, 12)
    assert get_dia_dos_namorados(2024) == date(2024, 6, 12)


# ----------------------------------------------- signed distance helper


def test_signed_distance_zero_on_event_day() -> None:
    """The day of the event itself has distance 0."""
    assert _signed_distance_to_nearest(date(2017, 11, 24), [date(2017, 11, 24)]) == 0


def test_signed_distance_negative_before_event() -> None:
    """Days before the event are negative."""
    assert _signed_distance_to_nearest(date(2017, 11, 20), [date(2017, 11, 24)]) == -4


def test_signed_distance_positive_after_event() -> None:
    """Days after the event are positive."""
    assert _signed_distance_to_nearest(date(2017, 11, 28), [date(2017, 11, 24)]) == 4


def test_signed_distance_picks_nearest_event() -> None:
    """When multiple events exist, return the signed distance to the nearest."""
    events = [date(2017, 11, 24), date(2018, 11, 23)]
    # Closer to 2017 event
    assert _signed_distance_to_nearest(date(2017, 12, 1), events) == 7
    # Closer to 2018 event
    assert _signed_distance_to_nearest(date(2018, 11, 20), events) == -3


# ----------------------------------------------------- EventFeatures core


@pytest.fixture
def bf_window_df() -> pd.DataFrame:
    """DataFrame covering the Black Friday 2017 window (Nov 20 - Nov 30).

    BF 2017 = Friday, November 24.
    Days from -4 to +6 around the event.
    """
    return pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(
                [
                    "2017-11-20",  # -4
                    "2017-11-21",  # -3
                    "2017-11-22",  # -2
                    "2017-11-23",  # -1
                    "2017-11-24",  # 0  (BF)
                    "2017-11-25",  # +1
                    "2017-11-26",  # +2
                    "2017-11-27",  # +3 (operational peak Mon)
                    "2017-11-28",  # +4 (operational peak Tue)
                    "2017-11-29",  # +5
                    "2017-11-30",  # +6
                ]
            ),
            "n_shipments": [195, 65, 150, 270, 325, 70, 0, 670, 707, 567, 487],
        }
    )


def test_days_to_black_friday_values(bf_window_df: pd.DataFrame) -> None:
    """days_to_black_friday must produce the exact signed distances."""
    builder = EventFeatures(track_dia_dos_namorados=False)
    out = builder.transform(bf_window_df)

    expected = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    assert out["days_to_black_friday"].tolist() == expected


def test_is_black_friday_window_default_7_days(bf_window_df: pd.DataFrame) -> None:
    """With default window 7/7, all rows in [-7, +7] should be flagged."""
    builder = EventFeatures(track_dia_dos_namorados=False)
    out = builder.transform(bf_window_df)

    # All 11 rows are within +/-7 of BF
    assert out["is_black_friday_window"].sum() == 11


def test_is_black_friday_window_narrower(bf_window_df: pd.DataFrame) -> None:
    """A narrower window must include fewer rows."""
    builder = EventFeatures(
        window_before=2,
        window_after=3,
        track_dia_dos_namorados=False,
    )
    out = builder.transform(bf_window_df)

    # In-window: days -2, -1, 0, 1, 2, 3 -> 6 rows
    assert out["is_black_friday_window"].sum() == 6


def test_post_black_friday_peak_marks_days_1_to_3(bf_window_df: pd.DataFrame) -> None:
    """Post-peak default of 3 days marks Nov 25, 26, 27 (days +1, +2, +3)."""
    builder = EventFeatures(track_dia_dos_namorados=False)
    out = builder.transform(bf_window_df)

    expected = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert out["is_post_black_friday_peak"].tolist() == expected


def test_dia_dos_namorados_features_added() -> None:
    """When track_dia_dos_namorados=True, three Dia dos Namorados columns appear."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(["2018-06-12"]),
            "n_shipments": [100],
        }
    )
    builder = EventFeatures(track_black_friday=False)
    out = builder.transform(df)

    assert "days_to_dia_dos_namorados" in out.columns
    assert out["days_to_dia_dos_namorados"].iloc[0] == 0  # event day itself
    assert out["is_dia_dos_namorados_window"].iloc[0] == 1


def test_multi_year_dataframe_uses_correct_event_per_year() -> None:
    """For a date in 2018, distance must be measured to 2018 BF, not 2017."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.to_datetime(
                [
                    "2017-11-24",  # 2017 BF: distance 0
                    "2018-11-23",  # 2018 BF: distance 0
                    "2017-11-30",  # 6 days after 2017 BF
                ]
            ),
            "n_shipments": [325, 400, 487],
        }
    )
    builder = EventFeatures(track_dia_dos_namorados=False)
    out = builder.transform(df)

    assert out["days_to_black_friday"].tolist() == [0, 0, 6]


def test_validation_rejects_no_tracked_events() -> None:
    """At least one event must be tracked."""
    with pytest.raises(ValueError, match="At least one event"):
        EventFeatures(track_black_friday=False, track_dia_dos_namorados=False)


def test_validation_rejects_negative_windows() -> None:
    """Negative window sizes must raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        EventFeatures(window_before=-1)
    with pytest.raises(ValueError, match="non-negative"):
        EventFeatures(window_after=-1)
    with pytest.raises(ValueError, match="non-negative"):
        EventFeatures(post_peak_days=-1)


def test_input_dataframe_is_not_mutated(bf_window_df: pd.DataFrame) -> None:
    """The transform must never modify the input DataFrame in place."""
    original_cols = bf_window_df.columns.tolist()
    EventFeatures().transform(bf_window_df)
    assert bf_window_df.columns.tolist() == original_cols


def test_feature_names_when_both_events_tracked() -> None:
    """feature_names must include all 6 columns (3 per event)."""
    builder = EventFeatures()
    assert builder.feature_names == [
        "days_to_black_friday",
        "is_black_friday_window",
        "is_post_black_friday_peak",
        "days_to_dia_dos_namorados",
        "is_dia_dos_namorados_window",
        "is_post_dia_dos_namorados_peak",
    ]


def test_feature_names_when_only_black_friday() -> None:
    """When only Black Friday is tracked, only 3 columns appear."""
    builder = EventFeatures(track_dia_dos_namorados=False)
    assert builder.feature_names == [
        "days_to_black_friday",
        "is_black_friday_window",
        "is_post_black_friday_peak",
    ]
