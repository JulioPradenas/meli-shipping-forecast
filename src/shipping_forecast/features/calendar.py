"""Calendar features for time-series forecasting.

Extracts deterministic features from the date column: day of week, month,
day of month, week of year, quarter. These features have no parameters
to learn; they exist to expose temporal regularities to the model.

Note on data types:
    All features are returned as integers (Python ``int``-like). Some
    models benefit from treating ``day_of_week`` and ``month`` as
    categorical at training time, but the conversion is the model's
    responsibility, not this builder's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from shipping_forecast.features.base import FeatureBuilder


@dataclass
class CalendarFeatures(FeatureBuilder):
    """Add calendar-based features derived from a date column.

    Supported features (all integers):

    * ``day_of_week``: 0 = Monday, 6 = Sunday
    * ``day_of_month``: 1-31
    * ``day_of_year``: 1-366
    * ``week_of_year``: 1-53
    * ``month``: 1-12
    * ``quarter``: 1-4
    * ``year``: e.g. 2017
    * ``is_weekend``: 1 if Saturday or Sunday else 0
    * ``is_month_start``: 1 if first day of the month
    * ``is_month_end``: 1 if last day of the month

    Attributes:
        features: Subset of the supported feature names to compute.
            Defaults to a sensible base set.
        sort_col: Column containing the date. Default ``shipment_date``.

    Raises:
        ValueError: If ``features`` is empty or contains unknown names.
    """

    features: list[str] = field(
        default_factory=lambda: [
            "day_of_week",
            "day_of_month",
            "month",
            "quarter",
            "is_weekend",
        ]
    )
    sort_col: str = "shipment_date"

    _SUPPORTED: ClassVar[set[str]] = {
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
    }

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("features must contain at least one element")
        invalid = set(self.features) - self._SUPPORTED
        if invalid:
            raise ValueError(
                f"Unsupported features: {sorted(invalid)}. Valid: {sorted(self._SUPPORTED)}"
            )

    @property
    def feature_names(self) -> list[str]:
        return list(self.features)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with calendar features added."""
        out = df.copy()
        dt = pd.to_datetime(out[self.sort_col]).dt

        extractors = {
            "day_of_week": lambda: dt.dayofweek.astype("int8"),
            "day_of_month": lambda: dt.day.astype("int8"),
            "day_of_year": lambda: dt.dayofyear.astype("int16"),
            "week_of_year": lambda: dt.isocalendar().week.astype("int8"),
            "month": lambda: dt.month.astype("int8"),
            "quarter": lambda: dt.quarter.astype("int8"),
            "year": lambda: dt.year.astype("int16"),
            "is_weekend": lambda: (dt.dayofweek >= 5).astype("int8"),
            "is_month_start": lambda: dt.is_month_start.astype("int8"),
            "is_month_end": lambda: dt.is_month_end.astype("int8"),
        }

        for feat in self.features:
            out[feat] = extractors[feat]()

        return out
