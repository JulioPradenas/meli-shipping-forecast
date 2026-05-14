"""Unit tests for forecasting metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shipping_forecast.evaluation.metrics import (
    bias,
    mae,
    rmse,
    wape,
    wape_by_segment,
    wape_in_event_window,
)

# --------------------------------------------------------------- point metrics


def test_wape_perfect_prediction() -> None:
    """When y_pred == y_true, WAPE must be 0."""
    y = [10.0, 20.0, 30.0]
    assert wape(y, y) == 0.0


def test_wape_known_example() -> None:
    """WAPE = sum(|residuals|) / sum(|truth|).

    Example: truth = [10, 20], pred = [12, 19]
        residuals = [2, 1] -> sum = 3
        truth abs sum = 30
        WAPE = 3/30 = 0.10
    """
    assert wape([10.0, 20.0], [12.0, 19.0]) == pytest.approx(0.10)


def test_wape_with_zeros_in_truth() -> None:
    """WAPE handles zero values without dividing by row-level zero.

    truth = [0, 0, 10], pred = [1, 1, 12]
        residuals = [1, 1, 2] -> sum = 4
        truth abs sum = 10
        WAPE = 4/10 = 0.40
    """
    assert wape([0.0, 0.0, 10.0], [1.0, 1.0, 12.0]) == pytest.approx(0.40)


def test_wape_raises_when_denominator_zero() -> None:
    """If sum(|y_true|) == 0, WAPE is undefined and must raise."""
    with pytest.raises(ValueError, match="undefined"):
        wape([0.0, 0.0], [1.0, 1.0])


def test_mae_perfect_prediction() -> None:
    assert mae([10.0, 20.0], [10.0, 20.0]) == 0.0


def test_mae_known_example() -> None:
    """MAE = mean(|residuals|). Example: errors of 2 and 4 -> MAE = 3."""
    assert mae([10.0, 20.0], [12.0, 24.0]) == pytest.approx(3.0)


def test_rmse_perfect_prediction() -> None:
    assert rmse([10.0, 20.0], [10.0, 20.0]) == 0.0


def test_rmse_known_example() -> None:
    """RMSE = sqrt(mean(squared residuals)).

    truth=[0, 0], pred=[3, 4] -> residuals=[3, 4]
        squared = [9, 16] -> mean = 12.5 -> sqrt ~ 3.535
    """
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_rmse_penalises_large_errors_more_than_mae() -> None:
    """Compared to MAE, RMSE punishes outliers.

    Same total error, but distributed differently:
        case A: errors of [3, 3]  -> MAE=3, RMSE=3
        case B: errors of [0, 6]  -> MAE=3, RMSE>3
    """
    case_a_mae = mae([0.0, 0.0], [3.0, 3.0])
    case_b_mae = mae([0.0, 0.0], [0.0, 6.0])
    case_a_rmse = rmse([0.0, 0.0], [3.0, 3.0])
    case_b_rmse = rmse([0.0, 0.0], [0.0, 6.0])

    assert case_a_mae == case_b_mae
    assert case_b_rmse > case_a_rmse


def test_bias_perfect_prediction() -> None:
    assert bias([10.0, 20.0], [10.0, 20.0]) == 0.0


def test_bias_over_prediction_is_positive() -> None:
    """When y_pred > y_true on average, bias must be > 0."""
    # truth = [10, 20], pred = [12, 22] -> errors = [+2, +2] -> bias = +2
    assert bias([10.0, 20.0], [12.0, 22.0]) == pytest.approx(2.0)


def test_bias_under_prediction_is_negative() -> None:
    """When y_pred < y_true on average, bias must be < 0."""
    assert bias([10.0, 20.0], [8.0, 18.0]) == pytest.approx(-2.0)


# --------------------------------------------------- input validation


def test_metrics_raise_on_empty_inputs() -> None:
    """All metrics must raise on empty arrays."""
    for fn in (wape, mae, rmse, bias):
        with pytest.raises(ValueError, match="empty"):
            fn([], [])


def test_metrics_raise_on_length_mismatch() -> None:
    """Inputs of different length must raise."""
    for fn in (wape, mae, rmse, bias):
        with pytest.raises(ValueError, match="same shape"):
            fn([1.0, 2.0], [1.0])


def test_metrics_accept_pandas_series() -> None:
    """Metrics must work with pandas Series, not just lists."""
    y_true = pd.Series([10.0, 20.0])
    y_pred = pd.Series([10.0, 20.0])
    assert wape(y_true, y_pred) == 0.0


def test_metrics_accept_numpy_arrays() -> None:
    """Metrics must work with numpy arrays."""
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 22.0])
    assert mae(y_true, y_pred) == pytest.approx(2.0)


# ----------------------------------------------- wape_by_segment


@pytest.fixture
def segmented_predictions_df() -> pd.DataFrame:
    """A small DataFrame with predictions for two states."""
    return pd.DataFrame(
        {
            "customer_state": ["SP", "SP", "RJ", "RJ"],
            "n_shipments": [100.0, 200.0, 50.0, 150.0],
            "y_pred": [110.0, 180.0, 55.0, 150.0],
        }
    )


def test_wape_by_segment_returns_one_value_per_segment(
    segmented_predictions_df: pd.DataFrame,
) -> None:
    """Each segment should produce exactly one WAPE value."""
    result = wape_by_segment(segmented_predictions_df, "customer_state")
    assert set(result.index) == {"SP", "RJ"}


def test_wape_by_segment_values_are_correct(
    segmented_predictions_df: pd.DataFrame,
) -> None:
    """Verify hand-calculated WAPE per segment.

    SP: truth = [100, 200], pred = [110, 180]
        residuals = [10, 20] -> sum = 30
        truth abs sum = 300
        WAPE = 30/300 = 0.10
    RJ: truth = [50, 150], pred = [55, 150]
        residuals = [5, 0] -> sum = 5
        truth abs sum = 200
        WAPE = 5/200 = 0.025
    """
    result = wape_by_segment(segmented_predictions_df, "customer_state")
    assert result.loc["SP"] == pytest.approx(0.10)
    assert result.loc["RJ"] == pytest.approx(0.025)


def test_wape_by_segment_drops_zero_truth_segments() -> None:
    """Segments where sum(|y_true|) == 0 must be excluded from the result."""
    df = pd.DataFrame(
        {
            "customer_state": ["SP", "RJ", "RJ"],
            "n_shipments": [100.0, 0.0, 0.0],
            "y_pred": [110.0, 5.0, 10.0],
        }
    )
    result = wape_by_segment(df, "customer_state")
    assert "SP" in result.index
    assert "RJ" not in result.index


def test_wape_by_segment_raises_on_missing_column(
    segmented_predictions_df: pd.DataFrame,
) -> None:
    """Missing required columns must raise KeyError with a helpful message."""
    with pytest.raises(KeyError, match="Missing column"):
        wape_by_segment(segmented_predictions_df, "nonexistent_col")


# ----------------------------------------------- wape_in_event_window


@pytest.fixture
def predictions_with_event_window_df() -> pd.DataFrame:
    """Predictions covering 10 days, with Black Friday 2017 in the middle."""
    return pd.DataFrame(
        {
            "shipment_date": pd.date_range("2017-11-20", periods=10),
            "n_shipments": [200.0, 220.0, 250.0, 280.0, 325.0, 70.0, 0.0, 670.0, 707.0, 567.0],
            "y_pred": [180.0, 210.0, 230.0, 260.0, 310.0, 80.0, 5.0, 600.0, 650.0, 540.0],
        }
    )


def test_wape_in_event_window_isolates_window_rows(
    predictions_with_event_window_df: pd.DataFrame,
) -> None:
    """The function must compute WAPE on rows within ±days_around of the event."""
    bf_2017 = pd.Timestamp("2017-11-24")
    result = wape_in_event_window(
        predictions_with_event_window_df,
        event_dates=[bf_2017],
        days_around=3,
    )
    # Should be a finite positive WAPE
    assert 0.0 < result < 1.0


def test_wape_in_event_window_raises_when_no_rows_match() -> None:
    """If no row falls in any event window, raise ValueError."""
    df = pd.DataFrame(
        {
            "shipment_date": pd.date_range("2017-01-01", periods=5),
            "n_shipments": [100.0] * 5,
            "y_pred": [100.0] * 5,
        }
    )
    # Event date far from data range
    with pytest.raises(ValueError, match="No rows"):
        wape_in_event_window(
            df,
            event_dates=[pd.Timestamp("2020-01-01")],
            days_around=3,
        )


def test_wape_in_event_window_handles_multiple_events(
    predictions_with_event_window_df: pd.DataFrame,
) -> None:
    """Multiple event dates should expand the window correctly."""
    # Two events: one inside the data, one outside. Union should still match.
    bf_2017 = pd.Timestamp("2017-11-24")
    fake_event = pd.Timestamp("2020-01-01")
    result = wape_in_event_window(
        predictions_with_event_window_df,
        event_dates=[bf_2017, fake_event],
        days_around=3,
    )
    assert result > 0
