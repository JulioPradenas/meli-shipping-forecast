"""Unit tests for time series cross-validation."""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pandas as pd
import pytest

from shipping_forecast.evaluation.cv import (
    DEFAULT_FOLDS,
    Fold,
    time_series_split,
)

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def olist_like_df() -> pd.DataFrame:
    """A DataFrame covering the modelable period of the Olist dataset.

    Two states, dense daily data from 2017-01-01 to 2018-08-31.
    """
    dates = pd.date_range("2017-01-01", "2018-08-31", freq="D")
    rows = []
    for state in ["SP", "RJ"]:
        for d in dates:
            rows.append(
                {
                    "shipment_date": d,
                    "customer_state": state,
                    "n_shipments": 100.0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------- DEFAULT_FOLDS


def test_default_folds_count() -> None:
    """The project default has exactly 4 folds."""
    assert len(DEFAULT_FOLDS) == 4


def test_default_folds_are_chronologically_ordered() -> None:
    """Each fold's test_start must be after the previous fold's test_start."""
    for prev, curr in pairwise(DEFAULT_FOLDS):
        assert curr["test_start"] > prev["test_start"]


def test_default_folds_test_start_after_train_end() -> None:
    """No leakage: test_start must be strictly after train_end in every fold."""
    for fold_spec in DEFAULT_FOLDS:
        assert fold_spec["test_start"] > fold_spec["train_end"]


# ----------------------------------------------------- time_series_split basic


def test_split_produces_default_4_folds(olist_like_df: pd.DataFrame) -> None:
    """Default behaviour returns exactly 4 folds."""
    folds = time_series_split(olist_like_df)
    assert len(folds) == 4


def test_each_fold_is_a_Fold_instance(olist_like_df: pd.DataFrame) -> None:
    """The function must return a list of Fold dataclass instances."""
    folds = time_series_split(olist_like_df)
    assert all(isinstance(f, Fold) for f in folds)


def test_fold_ids_are_sequential(olist_like_df: pd.DataFrame) -> None:
    """Folds must be numbered 1, 2, 3, 4."""
    folds = time_series_split(olist_like_df)
    assert [f.fold_id for f in folds] == [1, 2, 3, 4]


def test_train_period_starts_at_data_min_by_default(
    olist_like_df: pd.DataFrame,
) -> None:
    """Without explicit train_start, every fold starts at the data's earliest date."""
    folds = time_series_split(olist_like_df)
    for f in folds:
        assert f.train_period[0] == date(2017, 1, 1)


# ----------------------------------------------------- expanding window check


def test_expanding_window_property(olist_like_df: pd.DataFrame) -> None:
    """Each subsequent fold must have a larger or equal training set."""
    folds = time_series_split(olist_like_df)
    sizes = [f.n_train for f in folds]
    for prev, curr in pairwise(sizes):
        assert curr > prev, "expanding window should grow"


def test_train_end_strictly_before_test_start(olist_like_df: pd.DataFrame) -> None:
    """Critical: no temporal leakage between train and test in any fold."""
    folds = time_series_split(olist_like_df)
    for f in folds:
        train_max = f.train["shipment_date"].max().date()
        test_min = f.test["shipment_date"].min().date()
        assert train_max < test_min, f"Leakage in fold {f.fold_id}"


# ----------------------------------------------------- semantic content of folds


def test_fold_1_covers_black_friday_2017(olist_like_df: pd.DataFrame) -> None:
    """Fold 1's test period must include Black Friday 2017 (Nov 24)."""
    folds = time_series_split(olist_like_df)
    bf_2017 = pd.Timestamp("2017-11-24")
    test_dates = folds[0].test["shipment_date"]
    assert bf_2017 in test_dates.values


def test_fold_3_covers_dia_dos_namorados(olist_like_df: pd.DataFrame) -> None:
    """Fold 3's test period must include Dia dos Namorados (Jun 12)."""
    folds = time_series_split(olist_like_df)
    ddn_2018 = pd.Timestamp("2018-06-12")
    test_dates = folds[2].test["shipment_date"]
    assert ddn_2018 in test_dates.values


def test_fold_4_is_final_holdout(olist_like_df: pd.DataFrame) -> None:
    """Fold 4 tests the final two months: 2018-07 to 2018-08."""
    folds = time_series_split(olist_like_df)
    assert folds[3].test_period == (date(2018, 7, 1), date(2018, 8, 31))


# ----------------------------------------------------- helpers and edge cases


def test_fold_n_train_and_n_test_properties(olist_like_df: pd.DataFrame) -> None:
    """Convenience properties report row counts."""
    folds = time_series_split(olist_like_df)
    f = folds[0]
    assert f.n_train == len(f.train)
    assert f.n_test == len(f.test)


def test_split_raises_when_date_col_missing() -> None:
    """If the date column does not exist, raise KeyError."""
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(KeyError, match="date_col"):
        time_series_split(df, date_col="not_a_column")


def test_split_raises_when_fold_has_empty_train() -> None:
    """If a fold's train mask is empty, raise ValueError."""
    df = pd.DataFrame({"shipment_date": pd.to_datetime(["2018-08-15"])})
    # Default folds start with train data from 2017; no rows match.
    with pytest.raises(ValueError, match="empty training set"):
        time_series_split(df)


def test_split_raises_when_fold_has_empty_test() -> None:
    """If a fold's test mask is empty, raise ValueError."""
    df = pd.DataFrame({"shipment_date": pd.to_datetime(["2017-01-01"])})
    with pytest.raises(ValueError, match="empty test set"):
        time_series_split(df)


def test_custom_folds_spec_overrides_default() -> None:
    """Passing folds_spec replaces the default behaviour."""
    df = pd.DataFrame({"shipment_date": pd.date_range("2020-01-01", "2020-12-31", freq="D")})
    custom_spec = [
        {
            "train_end": date(2020, 6, 30),
            "test_start": date(2020, 7, 1),
            "test_end": date(2020, 12, 31),
            "label": "custom: 6mo train / 6mo test",
        }
    ]
    folds = time_series_split(df, folds_spec=custom_spec)
    assert len(folds) == 1
    assert folds[0].label == "custom: 6mo train / 6mo test"


def test_custom_folds_spec_rejects_overlapping_train_test() -> None:
    """A spec where test_start <= train_end must raise."""
    df = pd.DataFrame({"shipment_date": pd.date_range("2020-01-01", "2020-12-31", freq="D")})
    bad_spec = [
        {
            "train_end": date(2020, 7, 1),
            "test_start": date(2020, 7, 1),  # NOT strictly after
            "test_end": date(2020, 12, 31),
            "label": "bad",
        }
    ]
    with pytest.raises(ValueError, match="strictly after"):
        time_series_split(df, folds_spec=bad_spec)


def test_fold_dataframes_are_independent_copies(olist_like_df: pd.DataFrame) -> None:
    """Mutating a fold's DataFrame must not affect the original input."""
    folds = time_series_split(olist_like_df)
    original_cols = olist_like_df.columns.tolist()

    folds[0].train["new_col"] = 99

    assert olist_like_df.columns.tolist() == original_cols
