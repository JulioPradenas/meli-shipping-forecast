"""Forecasting evaluation utilities.

Public API:

* :func:`wape`, :func:`mae`, :func:`rmse`, :func:`bias`: point metrics.
* :func:`wape_by_segment`: WAPE grouped by an arbitrary segment column.
* :func:`wape_in_event_window`: WAPE restricted to a window around
  given event dates.
* :func:`asymmetric_cost`, :func:`expected_gain`,
  :func:`optimal_threshold_multiplier`: cost-sensitive metrics.
* :func:`time_series_split`: expanding-window cross-validation.
* :class:`Fold`: dataclass representing a single train/test split.
"""

from shipping_forecast.evaluation.cost_metrics import (
    asymmetric_cost,
    expected_gain,
    optimal_threshold_multiplier,
)
from shipping_forecast.evaluation.cv import DEFAULT_FOLDS, Fold, time_series_split
from shipping_forecast.evaluation.metrics import (
    bias,
    mae,
    rmse,
    wape,
    wape_by_segment,
    wape_in_event_window,
)

__all__ = [
    "DEFAULT_FOLDS",
    "Fold",
    "asymmetric_cost",
    "bias",
    "expected_gain",
    "mae",
    "optimal_threshold_multiplier",
    "rmse",
    "time_series_split",
    "wape",
    "wape_by_segment",
    "wape_in_event_window",
]
