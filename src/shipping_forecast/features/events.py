"""Commercial event features for Brazilian e-commerce forecasting.

Encodes the major commercial events identified during EDA:

* **Black Friday**: last Friday of November. The operational peak occurs
  3-4 business days **after** the event (Monday/Tuesday following),
  because customers buy on Friday but warehouses dispatch on Mon/Tue.
* **Dia dos Namorados**: June 12 every year. Brazilian Valentine's Day,
  the second-biggest commercial event after Black Friday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from shipping_forecast.features.base import FeatureBuilder
from shipping_forecast.utils.calendar_br import (
    get_black_friday,
    get_dia_dos_namorados,
)


@dataclass
class EventFeatures(FeatureBuilder):
    """Add commercial-event features per row.

    For each tracked event, three features are added:

    * ``days_to_{event}``: signed distance in days. Negative = before
      the event, 0 = the day of the event, positive = after.
    * ``is_{event}_window``: 1 if within the configured window before/after
      the event, else 0.
    * ``is_post_{event}_peak``: 1 for the 3 days immediately AFTER the
      event (captures the operational peak that happens after the sales).

    Attributes:
        window_before: Days before the event still considered "in window".
        window_after: Days after the event still considered "in window".
        post_peak_days: How many days after the event count as "peak".
        sort_col: Column containing the date. Default ``shipment_date``.
        track_black_friday: Whether to add Black Friday features.
        track_dia_dos_namorados: Whether to add Dia dos Namorados features.

    Raises:
        ValueError: If no events are tracked, or window sizes are negative.
    """

    window_before: int = 7
    window_after: int = 7
    post_peak_days: int = 3
    sort_col: str = "shipment_date"
    track_black_friday: bool = True
    track_dia_dos_namorados: bool = True

    def __post_init__(self) -> None:
        if not (self.track_black_friday or self.track_dia_dos_namorados):
            raise ValueError("At least one event must be tracked")
        if self.window_before < 0 or self.window_after < 0:
            raise ValueError(
                f"Window sizes must be non-negative; got "
                f"before={self.window_before}, after={self.window_after}"
            )
        if self.post_peak_days < 0:
            raise ValueError(f"post_peak_days must be non-negative; got {self.post_peak_days}")

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.track_black_friday:
            names += [
                "days_to_black_friday",
                "is_black_friday_window",
                "is_post_black_friday_peak",
            ]
        if self.track_dia_dos_namorados:
            names += [
                "days_to_dia_dos_namorados",
                "is_dia_dos_namorados_window",
                "is_post_dia_dos_namorados_peak",
            ]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with event features added."""
        out = df.copy()
        dates = pd.to_datetime(out[self.sort_col]).dt.date

        years = sorted({d.year for d in dates})

        if self.track_black_friday:
            event_dates = [get_black_friday(y) for y in years]
            self._add_event_features(out, dates, event_dates, "black_friday")

        if self.track_dia_dos_namorados:
            event_dates = [get_dia_dos_namorados(y) for y in years]
            self._add_event_features(out, dates, event_dates, "dia_dos_namorados")

        return out

    def _add_event_features(
        self,
        out: pd.DataFrame,
        dates: pd.Series,
        event_dates: list[date],
        prefix: str,
    ) -> None:
        """Compute features for a single event family.

        Mutates ``out`` by adding three columns for this event.
        """
        days_to = dates.apply(lambda d: _signed_distance_to_nearest(d, event_dates)).astype("int32")

        out[f"days_to_{prefix}"] = days_to

        in_window = (days_to >= -self.window_before) & (days_to <= self.window_after)
        out[f"is_{prefix}_window"] = in_window.astype("int8")

        in_post_peak = (days_to >= 1) & (days_to <= self.post_peak_days)
        out[f"is_post_{prefix}_peak"] = in_post_peak.astype("int8")


def _signed_distance_to_nearest(d: date, event_dates: list[date]) -> int:
    """Return the signed distance in days to the nearest event date.

    Sign convention: negative = event is in the future, positive = past.
    """
    deltas = [(d - ev).days for ev in event_dates]
    # Pick the one with smallest absolute value
    return min(deltas, key=abs)
