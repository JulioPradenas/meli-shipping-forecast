"""Feature engineering layer for shipping forecasting.

Public API:

* :class:`FeatureBuilder`: abstract base for all feature builders.
* :class:`LagFeatures`: shift-based lags per group.
* :class:`RollingFeatures`: rolling-window statistics per group.
* :class:`CalendarFeatures`: deterministic date-derived features.
* :class:`HolidayFeatures`: Brazilian operational calendar features.
* :class:`EventFeatures`: commercial event features (Black Friday, etc.).
* :class:`TrendFeatures`: long-term trend features.
* :class:`FeaturePipeline`: orchestrator to chain multiple builders.
"""

from shipping_forecast.features.base import FeatureBuilder
from shipping_forecast.features.calendar import CalendarFeatures
from shipping_forecast.features.events import EventFeatures
from shipping_forecast.features.holidays import HolidayFeatures
from shipping_forecast.features.lags import LagFeatures
from shipping_forecast.features.pipeline import FeaturePipeline
from shipping_forecast.features.rolling import RollingFeatures
from shipping_forecast.features.trend import TrendFeatures

__all__ = [
    "CalendarFeatures",
    "EventFeatures",
    "FeatureBuilder",
    "FeaturePipeline",
    "HolidayFeatures",
    "LagFeatures",
    "RollingFeatures",
    "TrendFeatures",
]
