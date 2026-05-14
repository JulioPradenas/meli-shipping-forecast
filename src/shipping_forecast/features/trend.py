"""Trend features for capturing long-term temporal dynamics.

The EDA revealed strong non-stationarity in the shipment volume:
~13x growth over 19 business months, with a structural break around
Black Friday 2017. Lag-based features alone cannot capture this — they
would require very long lags (months) which themselves suffer from
missing values at the start of the series.

This module exposes simple monotonic time features that let tree-based
models partition the series by era (early vs. mature vs. post-BF) and
discover non-linear trend effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from shipping_forecast.features.base import FeatureBuilder


@dataclass
class TrendFeatures(FeatureBuilder):
    """Add time-trend features anchored on a reference date.

    Adds three features:

    * ``days_since_start``: integer count of days from :attr:`reference_date`.
      Captures monotonic time progression.
    * ``month_index``: integer index that increases by 1 each calendar month
      starting at 0 for :attr:`reference_date`'s month. Useful as a
      categorical bucket in tree-based models.
    * ``year_progress``: position within the calendar year as a float in
      ``[0, 1)``. Captures within-year seasonality independently of which
      year it is.

    Attributes:
        reference_date: Anchor for ``days_since_start`` and ``month_index``.
            Defaults to ``date(2017, 1, 1)``, the start of the modelable
            period for the Olist dataset.
        sort_col: Column containing the date. Default ``shipment_date``.
        features: Which subset of trend features to add. Defaults to all.

    Raises:
        ValueError: If ``features`` is empty or contains unknown names.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "shipment_date": pd.to_datetime(["2017-01-01", "2017-02-01"]),
        ...     "n_shipments": [10, 20],
        ... })
        >>> out = TrendFeatures().transform(df)
        >>> out["days_since_start"].tolist()
        [0, 31]
        >>> out["month_index"].tolist()
        [0, 1]
    """

    reference_date: date = field(default_factory=lambda: date(2017, 1, 1))
    sort_col: str = "shipment_date"
    features: list[str] = field(
        default_factory=lambda: [
            "days_since_start",
            "month_index",
            "year_progress",
        ]
    )

    _SUPPORTED: tuple[str, ...] = (
        "days_since_start",
        "month_index",
        "year_progress",
    )

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("features must contain at least one element")
        invalid = set(self.features) - set(self._SUPPORTED)
        if invalid:
            raise ValueError(
                f"Unsupported features: {sorted(invalid)}. Valid: {list(self._SUPPORTED)}"
            )

    @property
    def feature_names(self) -> list[str]:
        return list(self.features)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with trend features added."""
        out = df.copy()
        dt = pd.to_datetime(out[self.sort_col])
        ref = pd.Timestamp(self.reference_date)

        if "days_since_start" in self.features:
            out["days_since_start"] = (dt - ref).dt.days.astype("int32")

        if "month_index" in self.features:
            # Months since reference_date (year-month difference)
            ref_idx = ref.year * 12 + ref.month
            row_idx = dt.dt.year * 12 + dt.dt.month
            out["month_index"] = (row_idx - ref_idx).astype("int16")

        if "year_progress" in self.features:
            # Position within the calendar year, in [0, 1)
            out["year_progress"] = (dt.dt.dayofyear - 1) / 365.0
            out["year_progress"] = out["year_progress"].astype("float32")

        return out
