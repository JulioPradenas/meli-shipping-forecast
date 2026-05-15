"""Forecasting models for shipping prediction.

Public API:

* :class:`ForecastModel`: abstract base for all forecasting models.
* :class:`NaiveForecaster`: baseline that predicts the last observed value.
* :class:`SeasonalNaiveForecaster`: baseline that predicts value from
  ``season`` days ago (default 7).
* :class:`ProphetForecaster`: additive decomposition model with built-in
  weekly/yearly seasonality and Brazilian operational calendar.
* :class:`LightGBMForecaster`: gradient-boosted trees over the canonical
  feature pipeline (Phase 6 production model).
"""

from shipping_forecast.models.base import ForecastModel
from shipping_forecast.models.lightgbm_model import LightGBMForecaster
from shipping_forecast.models.naive import NaiveForecaster
from shipping_forecast.models.prophet_model import ProphetForecaster
from shipping_forecast.models.seasonal_naive import SeasonalNaiveForecaster

__all__ = [
    "ForecastModel",
    "LightGBMForecaster",
    "NaiveForecaster",
    "ProphetForecaster",
    "SeasonalNaiveForecaster",
]
