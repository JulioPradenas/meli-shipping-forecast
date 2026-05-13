"""Rolling-window features for time-series forecasting.

Adds columns with rolling statistics (mean, std, min, max) over the
last N days, computed per group. All windows **exclude the current day**
to prevent label leakage during model training.

Leakage guard:
    A naive ``df.rolling(window=7).mean()`` includes the current row in
    the window, which leaks the target into its own predictors. We apply
    ``.shift(1)`` after the rolling op so that the window covers exactly
    [t-N, t-1], never touching t itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import pandas as pd

from shipping_forecast.features.base import FeatureBuilder

Stat = Literal["mean", "std", "min", "max"]


@dataclass
class RollingFeatures(FeatureBuilder):
    """Add rolling-window statistics per group.

    For every combination of ``window`` and ``stat``, adds a column named
    ``rolling_{stat}_{window}`` containing that statistic over the previous
    ``window`` days within the same group.

    Attributes:
        windows: Window sizes in days. Defaults to ``[7, 14, 28]``.
        stats: List of statistics to compute. Subset of
            ``{"mean", "std", "min", "max"}``. Defaults to ``["mean"]``.
        group_col: Column to partition by. Default ``customer_state``.
        sort_col: Column defining temporal order. Default ``shipment_date``.
        target_col: Column whose stats are computed. Default ``n_shipments``.
        min_periods: Minimum observations required to produce a value;
            otherwise NaN. Defaults to ``1`` (mean works with just 1 sample,
            std needs at least 2 and will return NaN until enough points).

    Raises:
        ValueError: If any window is non-positive, lists are empty, or
            any stat is not supported.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "customer_state": ["SP"] * 5,
        ...     "shipment_date": pd.date_range("2024-01-01", periods=5),
        ...     "n_shipments": [10, 20, 30, 40, 50],
        ... })
        >>> builder = RollingFeatures(windows=[2], stats=["mean"])
        >>> out = builder.transform(df)
        >>> out["rolling_mean_2"].tolist()
        [nan, 10.0, 15.0, 25.0, 35.0]
    """

    windows: list[int] = field(default_factory=lambda: [7, 14, 28])
    stats: list[Stat] = field(default_factory=lambda: ["mean"])
    group_col: str = "customer_state"
    sort_col: str = "shipment_date"
    target_col: str = "n_shipments"
    min_periods: int = 1

    _VALID_STATS: ClassVar[set[Stat]] = {"mean", "std", "min", "max"}

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("windows must contain at least one element")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"All windows must be positive; got {self.windows}")
        if not self.stats:
            raise ValueError("stats must contain at least one element")
        invalid = set(self.stats) - self._VALID_STATS
        if invalid:
            raise ValueError(
                f"Unsupported stats: {sorted(invalid)}. Valid: {sorted(self._VALID_STATS)}"
            )

    @property
    def feature_names(self) -> list[str]:
        return [f"rolling_{stat}_{w}" for w in self.windows for stat in self.stats]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with rolling features added.

        Strategy:
            1. Sort by (group, date) so rolling windows are temporal.
            2. For each (window, stat), compute the rolling op then shift
               by 1 within each group, so the window never includes t.
        """
        out = df.copy()
        out = out.sort_values([self.group_col, self.sort_col]).reset_index(drop=True)
        grouped = out.groupby(self.group_col, sort=False)[self.target_col]

        for w in self.windows:
            rolled = grouped.rolling(window=w, min_periods=self.min_periods)
            for stat in self.stats:
                col = f"rolling_{stat}_{w}"
                # Compute the rolling op, then strip the group index
                values = getattr(rolled, stat)().reset_index(level=0, drop=True)
                # Shift by 1 within each group to exclude the current day
                out[col] = values.groupby(out[self.group_col]).shift(1)

        return out
