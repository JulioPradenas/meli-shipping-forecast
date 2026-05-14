"""Seasonal Naive forecaster: predicts the value from k periods ago.

For weekly data, ``k=7`` is the standard choice: predict Tuesday from
last Tuesday, Saturday from last Saturday. This captures the entire
weekly seasonality pattern (peak on Tue, low on Sat, zero on Sun)
without any training.

Mathematically: y_pred(t + h) = y_train(t + h - season), where ``season``
is the chosen period (default 7 for weekly).

Purpose:
    Despite its simplicity, this is the *real* baseline to beat for
    retail-style forecasting. It captures all weekly seasonality for
    free, which is the dominant signal in our dataset. Any complex
    model that doesn't beat it by a clear margin isn't justifying its
    complexity.

Limitations:
    - Cannot capture trend (will predict the same level forever).
    - Cannot react to events (Black Friday peak will not be predicted
      unless the previous week was also Black Friday — which it wasn't).
    - For horizon > season, predictions repeat cyclically.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from shipping_forecast.models.base import ForecastModel


@dataclass
class SeasonalNaiveForecaster(ForecastModel):
    """Predict the value from ``season`` days ago per group.

    Attributes:
        season: Number of periods (days) used as seasonality. Defaults
            to 7 for weekly patterns.
        date_col: Date column name.
        target_col: Target variable column name.
        interval_width: Number of historical standard deviations to use
            for the prediction interval. Defaults to 1.28 (~80%).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "shipment_date": pd.date_range("2024-01-01", periods=14),
        ...     "customer_state": ["SP"] * 14,
        ...     "n_shipments": [100, 110, 120, 130, 140, 150, 160,
        ...                      200, 210, 220, 230, 240, 250, 260],
        ... })
        >>> model = SeasonalNaiveForecaster(season=7).fit(df)
        >>> preds = model.predict(df, horizon=7)
        >>> # Each forecast day uses the value from 7 days ago in train
        >>> preds["y_pred"].tolist()
        [200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0]
    """

    season: int = 7
    date_col: str = "shipment_date"
    target_col: str = "n_shipments"
    interval_width: float = 1.28

    # Runtime state populated by fit()
    _last_season_values: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _last_date: pd.Timestamp | None = field(default=None, init=False, repr=False)
    _residual_std: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    _trained_at: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.season < 1:
            raise ValueError(f"season must be >= 1; got {self.season}")

    def fit(self, df: pd.DataFrame, group_col: str = "customer_state") -> SeasonalNaiveForecaster:
        """Memorise the last ``season`` values per group plus residual std."""
        self._validate_columns(df, group_col)

        sorted_df = df.sort_values(self.date_col)
        self._last_date = pd.Timestamp(sorted_df[self.date_col].max())

        for group, group_df in sorted_df.groupby(group_col, observed=True):
            group_df = group_df.sort_values(self.date_col)
            values = group_df[self.target_col].tolist()
            if len(values) < self.season:
                raise ValueError(
                    f"Group {group!r} has only {len(values)} observations; "
                    f"need at least {self.season} for season={self.season}"
                )
            # Store the last `season` values in chronological order
            self._last_season_values[str(group)] = [float(v) for v in values[-self.season :]]
            # Residual std on seasonal differences for interval width
            seasonal_diffs = pd.Series(values).diff(self.season).dropna()
            self._residual_std[str(group)] = (
                float(seasonal_diffs.std()) if len(seasonal_diffs) > 1 else 0.0
            )

        self._fitted = True
        self._trained_at = datetime.now().isoformat(timespec="seconds")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        group_col: str = "customer_state",
    ) -> pd.DataFrame:
        """Forecast next ``horizon`` days by cycling the last ``season`` values."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")

        assert self._last_date is not None
        forecast_dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        rows: list[dict] = []
        for group, season_values in self._last_season_values.items():
            std = self._residual_std.get(group, 0.0)
            margin = self.interval_width * std
            for offset, forecast_date in enumerate(forecast_dates):
                # Cycle through season_values: day 0 takes value at index 0,
                # day 7 takes value at index 0 again, etc.
                value = season_values[offset % self.season]
                rows.append(
                    {
                        self.date_col: forecast_date,
                        group_col: group,
                        "y_pred": value,
                        "y_lower": max(0.0, value - margin),
                        "y_upper": value + margin,
                    }
                )

        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        """Persist model + JSON sidecar."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path.with_suffix(".joblib"))

        metadata = {
            "model_name": self.__class__.__name__,
            "trained_at": self._trained_at,
            "last_train_date": self._last_date.isoformat() if self._last_date else None,
            "n_groups": len(self._last_season_values),
            "groups": sorted(self._last_season_values.keys()),
            "params": {
                "season": self.season,
                "date_col": self.date_col,
                "target_col": self.target_col,
                "interval_width": self.interval_width,
            },
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> SeasonalNaiveForecaster:
        """Restore a model previously saved with :meth:`save`."""
        path = Path(path)
        model = joblib.load(path.with_suffix(".joblib"))
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is a {type(model).__name__}, expected {cls.__name__}")
        return model

    def _validate_columns(self, df: pd.DataFrame, group_col: str) -> None:
        for col in (self.date_col, group_col, self.target_col):
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col!r}")
