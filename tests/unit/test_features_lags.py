"""Unit tests for LagFeatures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shipping_forecast.features.lags import LagFeatures

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def small_df() -> pd.DataFrame:
    """A minimal time-series DataFrame with two groups (SP, RJ).

    Each group has 5 sequential days with known shipment counts so we can
    verify lag values arithmetically.
    """
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


def test_single_lag_happy_path(small_df: pd.DataFrame) -> None:
    """lag_1 must return the previous day's value within each group."""
    builder = LagFeatures(lags=[1])
    out = builder.transform(small_df)

    sp = out[out["customer_state"] == "SP"].sort_values("shipment_date")
    rj = out[out["customer_state"] == "RJ"].sort_values("shipment_date")

    # SP: [10, 20, 30, 40, 50] -> lag_1 = [NaN, 10, 20, 30, 40]
    assert sp["lag_1"].iloc[0] != sp["lag_1"].iloc[0]  # NaN check
    assert sp["lag_1"].iloc[1:].tolist() == [10.0, 20.0, 30.0, 40.0]

    # RJ: [5, 15, 25, 35, 45] -> lag_1 = [NaN, 5, 15, 25, 35]
    assert rj["lag_1"].iloc[0] != rj["lag_1"].iloc[0]  # NaN check
    assert rj["lag_1"].iloc[1:].tolist() == [5.0, 15.0, 25.0, 35.0]


def test_multiple_lags_simultaneous(small_df: pd.DataFrame) -> None:
    """Multiple lags must coexist and have independent NaN regions."""
    builder = LagFeatures(lags=[1, 3])
    out = builder.transform(small_df)

    sp = out[out["customer_state"] == "SP"].sort_values("shipment_date").reset_index(drop=True)

    # lag_1: NaN, 10, 20, 30, 40
    # lag_3: NaN, NaN, NaN, 10, 20
    assert pd.isna(sp["lag_1"].iloc[0])
    assert sp["lag_1"].iloc[4] == 40.0

    assert pd.isna(sp["lag_3"].iloc[2])
    assert sp["lag_3"].iloc[3] == 10.0
    assert sp["lag_3"].iloc[4] == 20.0


def test_no_temporal_leakage_when_unsorted_input() -> None:
    """Even if rows arrive in random order, lags must read past values only.

    This is the critical test: if the implementation forgets to sort, the
    lag could end up pointing to a *future* row, which is silent label
    leakage.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "customer_state": ["SP"] * 10,
            "shipment_date": pd.date_range("2024-01-01", periods=10),
            "n_shipments": list(range(100, 200, 10)),  # 100, 110, ..., 190
        }
    )
    # Shuffle the rows to simulate unsorted input
    shuffled = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    builder = LagFeatures(lags=[1])
    out = builder.transform(shuffled)

    # After transform, output should be sorted by date.
    # Compare lag_1 vs n_shipments shifted manually.
    out_sorted = out.sort_values("shipment_date").reset_index(drop=True)
    expected_lag = out_sorted["n_shipments"].shift(1)

    pd.testing.assert_series_equal(out_sorted["lag_1"], expected_lag, check_names=False)


def test_validation_rejects_empty_lags() -> None:
    """An empty list of lags must raise ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        LagFeatures(lags=[])


def test_validation_rejects_non_positive_lags() -> None:
    """Zero or negative lags must raise ValueError (would cause leakage)."""
    with pytest.raises(ValueError, match="positive"):
        LagFeatures(lags=[0])
    with pytest.raises(ValueError, match="positive"):
        LagFeatures(lags=[1, -1])


def test_input_dataframe_is_not_mutated(small_df: pd.DataFrame) -> None:
    """The transform must never modify the input DataFrame in place."""
    original_cols = small_df.columns.tolist()
    original_shape = small_df.shape

    LagFeatures(lags=[1, 7]).transform(small_df)

    assert small_df.columns.tolist() == original_cols
    assert small_df.shape == original_shape


def test_feature_names_property() -> None:
    """feature_names must return the column names that will be added."""
    builder = LagFeatures(lags=[1, 7, 28])
    assert builder.feature_names == ["lag_1", "lag_7", "lag_28"]


def test_repr_includes_lag_values() -> None:
    """The auto-generated dataclass __repr__ should mention the lag values."""
    builder = LagFeatures(lags=[1, 7])
    r = repr(builder)
    assert "LagFeatures" in r
    assert "1" in r
    assert "7" in r
