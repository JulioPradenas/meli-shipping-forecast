"""Cost-sensitive evaluation metrics.

The point-forecast metrics in :mod:`shipping_forecast.evaluation.metrics`
treat over- and under-prediction symmetrically. In logistics this is
rarely realistic: under-prediction (insufficient capacity, missed
deliveries, customer wait) typically costs 2-3x more per unit than
over-prediction (idle capacity, sunk fixed costs). These metrics expose
that asymmetry explicitly.

All functions in this module are **pure**: they take arrays and floats,
return floats. No state, no side effects, vectorised over numpy.

Cost convention
---------------
* ``c_under``: USD per shipment of under-prediction (y_pred < y_true).
* ``c_over``: USD per shipment of over-prediction (y_pred > y_true).
* The asymmetric cost on a single observation is::

      cost_i = c_under * max(0, y_true - y_pred)
             + c_over  * max(0, y_pred - y_true)

* Total cost is the sum over all observations.

Gain convention
---------------
``expected_gain(model, baseline)`` = ``total_cost(baseline) - total_cost(model)``.

A positive number means the model saves money compared to the baseline.
A negative number means the model is worse and costs more.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

ArrayLike = Iterable[float] | np.ndarray | pd.Series


def _to_array(values: ArrayLike) -> np.ndarray:
    """Convert any array-like to a 1-D float numpy array."""
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False)
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=float, copy=False)
    return np.array(list(values), dtype=float)


def asymmetric_cost(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    c_under: float,
    c_over: float,
) -> float:
    """Total asymmetric cost over a set of predictions.

    Args:
        y_true: Ground-truth shipment counts.
        y_pred: Predicted shipment counts.
        c_under: Cost per shipment of under-prediction (USD/shipment).
        c_over: Cost per shipment of over-prediction (USD/shipment).

    Returns:
        Total cost as a float (in USD if ``c_under`` and ``c_over`` are
        in USD). Lower is better.

    Raises:
        ValueError: If inputs are empty, mismatched in length, or
            either cost is negative.
    """
    if c_under < 0 or c_over < 0:
        raise ValueError(f"Costs must be non-negative; got c_under={c_under}, c_over={c_over}")

    yt = _to_array(y_true)
    yp = _to_array(y_pred)

    if yt.size == 0:
        raise ValueError("y_true is empty")
    if yt.size != yp.size:
        raise ValueError(f"Length mismatch: y_true has {yt.size}, y_pred has {yp.size}")

    errors = yt - yp
    under_pred = np.maximum(errors, 0.0)  # y_true > y_pred
    over_pred = np.maximum(-errors, 0.0)  # y_pred > y_true
    return float(c_under * under_pred.sum() + c_over * over_pred.sum())


def expected_gain(
    y_true: ArrayLike,
    y_pred_model: ArrayLike,
    y_pred_baseline: ArrayLike,
    c_under: float,
    c_over: float,
) -> float:
    """Expected gain in USD of a model over a baseline.

    A positive number means the model saves money vs the baseline.
    A negative number means the model costs more.

    Computed as::

        gain = total_cost(baseline) - total_cost(model)

    Args:
        y_true: Ground-truth shipment counts.
        y_pred_model: Predictions from the model being evaluated.
        y_pred_baseline: Predictions from the baseline (typically the
            simpler model we are trying to beat — SeasonalNaive here).
        c_under: Cost per shipment of under-prediction.
        c_over: Cost per shipment of over-prediction.

    Returns:
        Expected gain in USD. Sign convention: positive = model wins.

    Raises:
        ValueError: If lengths don't match or costs are negative.
    """
    cost_model = asymmetric_cost(y_true, y_pred_model, c_under, c_over)
    cost_baseline = asymmetric_cost(y_true, y_pred_baseline, c_under, c_over)
    return cost_baseline - cost_model


def optimal_threshold_multiplier(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    c_under: float,
    c_over: float,
    alpha_grid: tuple[float, ...] = (
        -0.20,
        -0.15,
        -0.10,
        -0.05,
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ),
) -> tuple[float, float]:
    """Find the multiplicative offset that minimises asymmetric cost.

    For each ``alpha`` in ``alpha_grid``, scales predictions by
    ``(1 + alpha)`` and evaluates the resulting cost. Returns the
    best ``alpha`` and its cost.

    This is a simple heuristic for cost-sensitive prediction. When
    under-prediction is more expensive than over-prediction, the optimal
    offset is positive (predict higher than the raw model output);
    inverse when over-prediction is more expensive.

    Args:
        y_true: Ground-truth shipment counts.
        y_pred: Raw model predictions.
        c_under: Cost per shipment of under-prediction.
        c_over: Cost per shipment of over-prediction.
        alpha_grid: Multipliers to test. The default covers a reasonable
            range from -10% to +50% in granular steps.

    Returns:
        ``(best_alpha, best_cost)`` tuple. ``best_alpha=0`` means no
        adjustment is optimal.

    Raises:
        ValueError: If ``alpha_grid`` is empty or inputs are invalid.
    """
    if not alpha_grid:
        raise ValueError("alpha_grid must be non-empty")

    yt = _to_array(y_true)
    yp = _to_array(y_pred)

    best_alpha = 0.0
    best_cost = float("inf")
    for alpha in alpha_grid:
        # Clip to non-negative (predictions cannot be negative)
        adjusted = np.maximum(yp * (1.0 + alpha), 0.0)
        cost = asymmetric_cost(yt, adjusted, c_under, c_over)
        if cost < best_cost:
            best_cost = cost
            best_alpha = float(alpha)

    return best_alpha, best_cost
