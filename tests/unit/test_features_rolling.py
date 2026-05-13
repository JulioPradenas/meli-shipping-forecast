"""Unit tests for RollingFeatures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shipping_forecast.features.rolling import RollingFeatures

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def small_df() -> pd.DataFrame:
    """A minimal DataFrame with two groups and known values for verification."""
    return pd.DataFrame(
        {
            "customer_state": ["SP"] * 5 + ["RJ"] * 5,
            "shipment_date": pd.concat(
                [pd.Series(pd.date_range("2024-01-01", periods=5))] * 2
            ).reset_index(drop=True),
            "n_shipments": [10, 20, 30, 40, 50, 5, 15, 25, 35, 45],
        }
    )


# --------------------------------------------------------------------- tests


def test_rolling_mean_excludes_current_day(small_df: pd.DataFrame) -> None:
    """The rolling mean must exclude the current day (no leakage).

    For SP with values [10, 20, 30, 40, 50] and window=2:
      day 0: window = []          -> NaN (no prior days)
      day 1: window = [10]        -> 10.0
      day 2: window = [10, 20]    -> 15.0
      day 3: window = [20, 30]    -> 25.0
      day 4: window = [30, 40]    -> 35.0
    """
    builder = RollingFeatures(windows=[2], stats=["mean"])
    out = builder.transform(small_df)

    sp = out[out["customer_state"] == "SP"].sort_values("shipment_date").reset_index(drop=True)

    assert pd.isna(sp["rolling_mean_2"].iloc[0])
    assert sp["rolling_mean_2"].iloc[1] == 10.0
    assert sp["rolling_mean_2"].iloc[2] == 15.0
    assert sp["rolling_mean_2"].iloc[3] == 25.0
    assert sp["rolling_mean_2"].iloc[4] == 35.0


def test_multiple_stats_simultaneous(small_df: pd.DataFrame) -> None:
    """Multiple statistics on the same window produce independent columns."""
    builder = RollingFeatures(windows=[3], stats=["mean", "max"])
    out = builder.transform(small_df)

    sp = out[out["customer_state"] == "SP"].sort_values("shipment_date").reset_index(drop=True)

    # Day 3 (value 40): window = [10, 20, 30] -> mean=20, max=30
    assert sp["rolling_mean_3"].iloc[3] == 20.0
    assert sp["rolling_max_3"].iloc[3] == 30.0


def test_groups_are_independent(small_df: pd.DataFrame) -> None:
    """Rolling stats must not bleed across groups (SP rolling != affected by RJ)."""
    builder = RollingFeatures(windows=[2], stats=["mean"])
    out = builder.transform(small_df)

    sp = out[out["customer_state"] == "SP"].sort_values("shipment_date").reset_index(drop=True)
    rj = out[out["customer_state"] == "RJ"].sort_values("shipment_date").reset_index(drop=True)

    # Both groups should have NaN on day 0 (no prior history)
    assert pd.isna(sp["rolling_mean_2"].iloc[0])
    assert pd.isna(rj["rolling_mean_2"].iloc[0])

    # RJ values [5, 15, 25, 35, 45]: day 2 with window=2 -> mean of [5, 15] = 10
    assert rj["rolling_mean_2"].iloc[2] == 10.0


def test_no_temporal_leakage_when_unsorted_input() -> None:
    """Critical test: even with shuffled input, rolling must look only at past."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "customer_state": ["SP"] * 10,
            "shipment_date": pd.date_range("2024-01-01", periods=10),
            "n_shipments": list(range(100, 200, 10)),
        }
    )
    shuffled = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    builder = RollingFeatures(windows=[3], stats=["mean"])
    out = builder.transform(shuffled)
    out_sorted = out.sort_values("shipment_date").reset_index(drop=True)

    # Day 4 (value 140): window = [110, 120, 130] -> mean = 120
    assert out_sorted["rolling_mean_3"].iloc[4] == 120.0


def test_validation_rejects_empty_windows() -> None:
    """Empty windows list must raise ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        RollingFeatures(windows=[])


def test_validation_rejects_non_positive_windows() -> None:
    """Zero or negative windows must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        RollingFeatures(windows=[0])
    with pytest.raises(ValueError, match="positive"):
        RollingFeatures(windows=[7, -3])


def test_validation_rejects_unsupported_stat() -> None:
    """Stats outside the whitelist must raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported stats"):
        RollingFeatures(stats=["sum"])  # type: ignore[list-item]


def test_input_dataframe_is_not_mutated(small_df: pd.DataFrame) -> None:
    """The transform must never modify the input DataFrame in place."""
    original_cols = small_df.columns.tolist()
    original_shape = small_df.shape

    RollingFeatures(windows=[3], stats=["mean", "std"]).transform(small_df)

    assert small_df.columns.tolist() == original_cols
    assert small_df.shape == original_shape


def test_feature_names_property() -> None:
    """feature_names lists every (window, stat) combination."""
    builder = RollingFeatures(windows=[7, 14], stats=["mean", "std"])
    assert set(builder.feature_names) == {
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_std_14",
    }
