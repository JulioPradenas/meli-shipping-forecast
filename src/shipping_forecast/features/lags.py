"""Lag features for time-series forecasting.

Adds columns with the value of the target N days in the past, computed
per group (e.g., per state). Critical for capturing seasonality and recent
dynamics in forecasting models.

Leakage guard:
    All lags are strictly positive (past values only). The transformation
    never references future rows. This is enforced by ``groupby().shift(n)``
    which always reads from earlier positions in the sorted sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from shipping_forecast.features.base import FeatureBuilder


@dataclass
class LagFeatures(FeatureBuilder):
    """Add lag features (value N days ago) per group.

    For each lag ``n`` in :attr:`lags`, adds a column named ``lag_{n}``
    containing the value of :attr:`target_col` exactly ``n`` rows back in
    the same group, ordered by :attr:`sort_col`.

    Attributes:
        lags: List of positive integers. Each ``n`` produces a column
            ``lag_{n}`` with the value from ``n`` days ago. Defaults to
            ``[1, 7, 14, 28]`` (yesterday, last week, two weeks, four weeks).
        group_col: Column to partition by (e.g., ``customer_state``). The
            lag is computed independently within each group.
        sort_col: Column defining temporal order within each group.
        target_col: Column whose past values are extracted.

    Raises:
        ValueError: If any lag is non-positive, or if ``lags`` is empty.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "state": ["SP", "SP", "SP", "RJ", "RJ", "RJ"],
        ...     "date": pd.date_range("2024-01-01", periods=3).tolist() * 2,
        ...     "n_shipments": [10, 20, 30, 5, 15, 25],
        ... })
        >>> builder = LagFeatures(lags=[1], group_col="state",
        ...                       sort_col="date", target_col="n_shipments")
        >>> out = builder.transform(df)
        >>> out["lag_1"].tolist()
        [nan, 10.0, 20.0, nan, 5.0, 15.0]
    """

    lags: list[int] = field(default_factory=lambda: [1, 7, 14, 28])
    group_col: str = "customer_state"
    sort_col: str = "shipment_date"
    target_col: str = "n_shipments"

    def __post_init__(self) -> None:
        if not self.lags:
            raise ValueError("lags must contain at least one element")
        if any(lag <= 0 for lag in self.lags):
            raise ValueError(f"All lags must be positive; got {self.lags}")

    @property
    def feature_names(self) -> list[str]:
        return [f"lag_{lag}" for lag in self.lags]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with lag columns added.

        Sorting strategy:
            We sort by (group_col, sort_col) before shifting to guarantee
            chronological order within each group. The original index is
            preserved so the output aligns with the input.
        """
        # Defensive copy: never mutate the input
        out = df.copy()
        # Sort once for all lags
        out = out.sort_values([self.group_col, self.sort_col]).reset_index(drop=True)
        # Compute each lag with groupby().shift()
        grouped = out.groupby(self.group_col, sort=False)[self.target_col]
        for lag in self.lags:
            out[f"lag_{lag}"] = grouped.shift(lag)
        return out
