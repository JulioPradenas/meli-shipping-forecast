"""Forecasting evaluation utilities.

Public API:

* :func:`wape`, :func:`mae`, :func:`rmse`, :func:`bias`: point metrics.
* :func:`wape_by_segment`: WAPE grouped by an arbitrary segment column.
* :func:`wape_in_event_window`: WAPE restricted to a window around
  given event dates.
* :func:`time_series_split`: expanding-window cross-validation.
* :class:`Fold`: dataclass representing a single train/test split.
"""

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
    "bias",
    "mae",
    "rmse",
    "time_series_split",
    "wape",
    "wape_by_segment",
    "wape_in_event_window",
]
