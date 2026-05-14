"""Forecasting evaluation metrics.

All point-forecast metrics in this module are **pure functions**: they
take true and predicted arrays and return a float. They are robust to
input types (lists, numpy arrays, pandas Series) and to empty inputs
(raise explicitly rather than returning NaN silently).

Primary metric: WAPE (weighted absolute percentage error). It is robust
to zero values (common in our dataset due to Sundays and holidays) and
gives a single percentage interpretable by stakeholders.

Convention:
    Throughout this module, residuals are computed as ``y_pred - y_true``
    so positive bias = systematic over-prediction.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Type alias for anything we can convert to a 1-D numpy array of floats
ArrayLike = Iterable[float] | np.ndarray | pd.Series


def _to_array(values: ArrayLike) -> np.ndarray:
    """Convert any array-like to a 1-D float numpy array.

    Raises:
        ValueError: If the input is empty.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute metric on empty input")
    return arr


def _validate_same_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {y_true.shape} and {y_pred.shape}"
        )


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted Absolute Percentage Error.

    .. math:: \\mathrm{WAPE} = \\frac{\\sum |y_i - \\hat y_i|}{\\sum |y_i|}

    Robust to zero values (uses sum of absolutes in denominator instead
    of per-row division). Returns a fraction (0.15 = 15%).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        WAPE as a float. Lower is better.

    Raises:
        ValueError: If inputs are empty, mismatched in length, or
            ``sum(|y_true|) == 0``.
    """
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    _validate_same_length(yt, yp)

    denominator = float(np.sum(np.abs(yt)))
    if denominator == 0:
        raise ValueError("WAPE is undefined when sum(|y_true|) == 0")
    return float(np.sum(np.abs(yt - yp)) / denominator)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error.

    .. math:: \\mathrm{MAE} = \\frac{1}{n} \\sum |y_i - \\hat y_i|

    Interpretable in the original units (e.g., "the model is off by
    34 packages per day on average").
    """
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    _validate_same_length(yt, yp)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error.

    .. math:: \\mathrm{RMSE} = \\sqrt{\\frac{1}{n} \\sum (y_i - \\hat y_i)^2}

    Penalises large errors more than small ones. Useful for logistics
    where one big under-forecast is worse than many small ones.
    """
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    _validate_same_length(yt, yp)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean error (positive = systematic over-prediction).

    .. math:: \\mathrm{Bias} = \\frac{1}{n} \\sum (\\hat y_i - y_i)

    Diagnostic metric: lets you check if the model is consistently
    over-predicting (bias > 0) or under-predicting (bias < 0). Ideally
    close to zero.
    """
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    _validate_same_length(yt, yp)
    return float(np.mean(yp - yt))


def wape_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    y_true_col: str = "n_shipments",
    y_pred_col: str = "y_pred",
) -> pd.Series:
    """Compute WAPE within each segment.

    Useful to diagnose where the model fails: per state, per day-type,
    per month, etc.

    Args:
        df: DataFrame with at least ``segment_col``, ``y_true_col``,
            ``y_pred_col``.
        segment_col: Column to group by.
        y_true_col: Name of the true-value column.
        y_pred_col: Name of the predicted-value column.

    Returns:
        A pandas Series indexed by segment, with WAPE per group.
        Segments where ``sum(|y_true|) == 0`` are excluded.
    """
    for col in (segment_col, y_true_col, y_pred_col):
        if col not in df.columns:
            raise KeyError(f"Missing column: {col!r}")

    # Vectorised computation: groupby().sum() of absolute residuals
    # divided by groupby().sum() of absolute truths. Much faster than apply.
    abs_residuals = (df[y_true_col] - df[y_pred_col]).abs()
    abs_truths = df[y_true_col].abs()

    grouped = (
        pd.DataFrame(
            {
                "abs_residual": abs_residuals,
                "abs_truth": abs_truths,
                "segment": df[segment_col],
            }
        )
        .groupby("segment", observed=True)
        .sum()
    )

    # Drop segments where denominator is zero (WAPE undefined)
    grouped = grouped[grouped["abs_truth"] > 0]

    result = grouped["abs_residual"] / grouped["abs_truth"]
    result.name = "wape"
    return result


def wape_in_event_window(
    df: pd.DataFrame,
    event_dates: Iterable[pd.Timestamp],
    days_around: int = 3,
    date_col: str = "shipment_date",
    y_true_col: str = "n_shipments",
    y_pred_col: str = "y_pred",
) -> float:
    """Compute WAPE restricted to a window around given event dates.

    Useful to check model behaviour specifically during commercial events
    like Black Friday or Dia dos Namorados.

    Args:
        df: DataFrame with the prediction results.
        event_dates: Iterable of event dates (timestamps).
        days_around: Half-width of the window. ``days_around=3`` means
            [event - 3, event + 3] inclusive.
        date_col: Date column.
        y_true_col: True-value column.
        y_pred_col: Predicted-value column.

    Returns:
        WAPE on the rows falling inside any event window.

    Raises:
        ValueError: If no rows fall in any event window.
    """
    for col in (date_col, y_true_col, y_pred_col):
        if col not in df.columns:
            raise KeyError(f"Missing column: {col!r}")

    dates = pd.to_datetime(df[date_col])
    event_ts = [pd.Timestamp(ev) for ev in event_dates]

    mask = pd.Series(False, index=df.index)
    delta = pd.Timedelta(days=days_around)
    for ev in event_ts:
        mask |= (dates >= ev - delta) & (dates <= ev + delta)

    if not mask.any():
        raise ValueError("No rows fall inside any event window")

    return wape(df.loc[mask, y_true_col], df.loc[mask, y_pred_col])
