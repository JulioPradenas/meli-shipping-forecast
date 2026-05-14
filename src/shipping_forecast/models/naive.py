"""Naive forecaster: predicts the last observed value per group.

This is the simplest possible baseline. For each group (state), it
predicts that every future day will equal the most recent observation
in the training data.

Mathematically: y_pred(t + h) = y_train(t_last) for all h >= 1.

Purpose:
    Establishes the *floor* of model performance. Any production model
    must beat this baseline by a meaningful margin to justify its
    additional complexity. If a complex model only marginally beats
    Naive, it's evidence that either:
      * The problem is fundamentally hard.
      * The features don't carry useful signal.
      * The model is over-engineered.

Limitations:
    Ignores trend, seasonality, events, and within-group dynamics.
    Expected to perform poorly on weekly patterns (will predict the
    same value for Tuesday and Saturday alike).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from shipping_forecast.models.base import ForecastModel


@dataclass
class NaiveForecaster(ForecastModel):
    """Predict the last observed value per group.

    Attributes:
        date_col: Column containing dates in the input data.
        target_col: Column containing the target variable.
        interval_width: Number of historical standard deviations to use
            when constructing the prediction interval. Defaults to 1.28
            (corresponds to ~80% confidence under a normal approximation).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "shipment_date": pd.date_range("2024-01-01", periods=10),
        ...     "customer_state": ["SP"] * 10,
        ...     "n_shipments": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        ... })
        >>> model = NaiveForecaster().fit(df)
        >>> preds = model.predict(df, horizon=2)
        >>> preds["y_pred"].tolist()
        [190.0, 190.0]
    """

    date_col: str = "shipment_date"
    target_col: str = "n_shipments"
    interval_width: float = 1.28  # ~80% under normal approximation

    # State set by fit() — populated runtime, not by the user
    _last_values: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_date: pd.Timestamp | None = field(default=None, init=False, repr=False)
    _residual_std: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    _trained_at: str | None = field(default=None, init=False, repr=False)

    def fit(self, df: pd.DataFrame, group_col: str = "customer_state") -> NaiveForecaster:
        """Memorise the last value per group and historical residual std.

        Args:
            df: Historical DataFrame with date, group and target columns.
            group_col: Column identifying separate series.

        Returns:
            ``self``.
        """
        self._validate_columns(df, group_col)

        sorted_df = df.sort_values(self.date_col)
        self._last_date = pd.Timestamp(sorted_df[self.date_col].max())

        # For each group: store last value and the std of day-to-day changes
        # (used to size the prediction interval).
        for group, group_df in sorted_df.groupby(group_col, observed=True):
            group_df = group_df.sort_values(self.date_col)
            self._last_values[str(group)] = float(group_df[self.target_col].iloc[-1])
            # Residual = day-to-day differences. The std of these residuals
            # gives a sensible interval width even for trivial models.
            diffs = group_df[self.target_col].diff().dropna()
            self._residual_std[str(group)] = float(diffs.std()) if len(diffs) > 1 else 0.0

        self._fitted = True
        self._trained_at = datetime.now().isoformat(timespec="seconds")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        group_col: str = "customer_state",
    ) -> pd.DataFrame:
        """Forecast the next ``horizon`` days per group.

        Each group gets ``horizon`` rows, all with the same ``y_pred``
        equal to that group's last training value.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")

        # Forecast dates: the day after the last training date, for `horizon` days
        assert self._last_date is not None  # for mypy
        forecast_dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        rows: list[dict] = []
        for group, last_value in self._last_values.items():
            std = self._residual_std.get(group, 0.0)
            margin = self.interval_width * std
            for forecast_date in forecast_dates:
                rows.append(
                    {
                        self.date_col: forecast_date,
                        group_col: group,
                        "y_pred": last_value,
                        "y_lower": max(0.0, last_value - margin),
                        "y_upper": last_value + margin,
                    }
                )

        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        """Persist model to ``{path}.joblib`` and metadata to ``{path}.json``."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path.with_suffix(".joblib"))

        metadata = {
            "model_name": self.__class__.__name__,
            "trained_at": self._trained_at,
            "last_train_date": self._last_date.isoformat() if self._last_date else None,
            "n_groups": len(self._last_values),
            "groups": sorted(self._last_values.keys()),
            "params": {
                "date_col": self.date_col,
                "target_col": self.target_col,
                "interval_width": self.interval_width,
            },
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> NaiveForecaster:
        """Restore a previously saved model from ``{path}.joblib``."""
        path = Path(path)
        model = joblib.load(path.with_suffix(".joblib"))
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is a {type(model).__name__}, expected {cls.__name__}")
        return model

    def _validate_columns(self, df: pd.DataFrame, group_col: str) -> None:
        for col in (self.date_col, group_col, self.target_col):
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col!r}")
