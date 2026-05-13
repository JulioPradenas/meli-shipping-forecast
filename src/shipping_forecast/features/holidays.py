"""Holiday and day-type features for the Brazilian operational calendar.

Encodes the operational structure of Brazilian shipping logistics
identified during EDA:

* ``sunday``: 0 shipments by design (no warehouse/carrier operation).
* ``holiday``: federal holiday or Carnival/Corpus Christi (~0 shipments).
* ``saturday``: ~8% of weekday volume (reduced operation).
* ``business_day``: full operation Monday-Friday non-holiday.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from shipping_forecast.features.base import FeatureBuilder
from shipping_forecast.utils.calendar_br import build_br_operational_holidays


@dataclass
class HolidayFeatures(FeatureBuilder):
    """Add Brazilian operational calendar features.

    Adds the following columns:

    * ``is_holiday``: 1 if the date is a Brazilian non-operational
      holiday (federal + Carnival + Corpus Christi), else 0.
    * ``is_business_day``: 1 if Monday-Friday and not a holiday, else 0.
    * ``is_saturday``: 1 if Saturday, else 0.
    * ``is_sunday``: 1 if Sunday, else 0.
    * ``is_operational``: 1 if business day or Saturday (any operation
      possible), else 0.
    * ``day_type``: string label in
      ``{'business_day', 'saturday', 'sunday', 'holiday'}``.

    Attributes:
        sort_col: Column containing the date. Default ``shipment_date``.

    Note:
        ``day_type`` is the only string column. The booleans are useful
        for filtering and linear models; ``day_type`` is useful as a
        categorical feature in tree-based models like LightGBM.
    """

    sort_col: str = "shipment_date"

    @property
    def feature_names(self) -> list[str]:
        return [
            "is_holiday",
            "is_business_day",
            "is_saturday",
            "is_sunday",
            "is_operational",
            "day_type",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with day-type features added."""
        out = df.copy()
        dates = pd.to_datetime(out[self.sort_col])

        # Build the operational holiday set covering the date range
        years = range(dates.dt.year.min(), dates.dt.year.max() + 1)
        op_holidays = build_br_operational_holidays(years)

        # Compute features
        date_objs = dates.dt.date
        is_holiday = date_objs.isin(op_holidays)
        dow = dates.dt.dayofweek

        is_sunday = dow == 6
        is_saturday = dow == 5
        is_business_day = (dow < 5) & ~is_holiday
        is_operational = is_business_day | is_saturday

        out["is_holiday"] = is_holiday.astype("int8")
        out["is_business_day"] = is_business_day.astype("int8")
        out["is_saturday"] = is_saturday.astype("int8")
        out["is_sunday"] = is_sunday.astype("int8")
        out["is_operational"] = is_operational.astype("int8")

        # day_type as ordered categorical for predictable label encoding
        day_type = pd.Series("business_day", index=out.index, dtype="object")
        day_type[is_saturday] = "saturday"
        day_type[is_sunday] = "sunday"
        day_type[is_holiday] = "holiday"  # holidays override day-of-week
        out["day_type"] = day_type.astype(
            pd.CategoricalDtype(categories=["business_day", "saturday", "sunday", "holiday"])
        )

        return out
