"""Prophet-based forecaster: additive decomposition model.

Wraps the open-source Prophet library (developed by Meta) to fit our
ForecastModel interface. Prophet decomposes a time series into:

    y(t) = trend(t) + weekly_seasonality(t) + yearly_seasonality(t)
           + holidays(t) + noise

Strengths for our problem:
    - Captures weekly seasonality natively (the dominant signal in our
      dataset).
    - Handles trend changes via a piecewise-linear formulation, which
      should accommodate the Black Friday 2017 regime change.
    - Built-in support for holidays, including the Brazilian operational
      calendar (federal + Carnival + Corpus Christi).
    - Returns confidence intervals out of the box (no manual sizing
      needed).

Limitations:
    - One model per group: Prophet doesn't natively pool information
      across series. We fit one Prophet model per state. Acceptable for
      27 states, but doesn't scale to thousands.
    - Doesn't use the engineered features (lags, rolling stats) from
      Phase 4. Those will benefit LightGBM in Phase 6.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from prophet import Prophet  # type: ignore[import-untyped]

from shipping_forecast.models.base import ForecastModel
from shipping_forecast.utils.calendar_br import build_br_operational_holidays


@dataclass
class ProphetForecaster(ForecastModel):
    """One Prophet model per group, with optional Brazilian holidays.

    Attributes:
        date_col: Column containing dates.
        target_col: Column containing the target variable.
        use_br_holidays: If True, feeds the Brazilian operational calendar
            (federal + Carnival + Corpus Christi) to Prophet as holidays.
        weekly_seasonality: Whether to include weekly seasonality.
        yearly_seasonality: Whether to include yearly seasonality. Set to
            False if you have less than a year of data per group.
        interval_width: Confidence interval width (e.g., 0.80 = 80%).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "shipment_date": pd.date_range("2017-01-01", periods=400),
        ...     "customer_state": ["SP"] * 400,
        ...     "n_shipments": [100 + i for i in range(400)],
        ... })
        >>> model = ProphetForecaster().fit(df)
        >>> preds = model.predict(df, horizon=7)
        >>> "y_pred" in preds.columns
        True
    """

    date_col: str = "shipment_date"
    target_col: str = "n_shipments"
    use_br_holidays: bool = True
    weekly_seasonality: bool = True
    yearly_seasonality: bool = True
    interval_width: float = 0.80

    # Runtime state populated by fit()
    _models: dict[str, Prophet] = field(default_factory=dict, init=False, repr=False)
    _last_date: pd.Timestamp | None = field(default=None, init=False, repr=False)
    _group_col: str | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    _trained_at: str | None = field(default=None, init=False, repr=False)

    def fit(self, df: pd.DataFrame, group_col: str = "customer_state") -> ProphetForecaster:
        """Train one Prophet model per group.

        Prophet expects columns named ``ds`` (date) and ``y`` (target),
        so we rename internally. Holidays are passed as a DataFrame with
        columns ``holiday`` and ``ds``.
        """
        self._validate_columns(df, group_col)

        sorted_df = df.sort_values(self.date_col)
        self._last_date = pd.Timestamp(sorted_df[self.date_col].max())
        self._group_col = group_col

        # Build holidays DataFrame once (shared across groups)
        holidays_df = self._build_holidays_df(df) if self.use_br_holidays else None

        for group, group_df in sorted_df.groupby(group_col, observed=True):
            prophet_df = group_df[[self.date_col, self.target_col]].rename(
                columns={self.date_col: "ds", self.target_col: "y"}
            )

            model = Prophet(
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                daily_seasonality=False,  # we have daily granularity, not sub-daily
                interval_width=self.interval_width,
                holidays=holidays_df,
            )

            # Suppress Prophet's noisy stdout during fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_df)

            self._models[str(group)] = model

        self._fitted = True
        self._trained_at = datetime.now().isoformat(timespec="seconds")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        group_col: str = "customer_state",
    ) -> pd.DataFrame:
        """Forecast next ``horizon`` days per group, including intervals."""
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
        future_df = pd.DataFrame({"ds": forecast_dates})

        rows: list[dict] = []
        for group, model in self._models.items():
            forecast = model.predict(future_df)
            for _, row in forecast.iterrows():
                rows.append(
                    {
                        self.date_col: row["ds"],
                        group_col: group,
                        "y_pred": max(0.0, float(row["yhat"])),
                        "y_lower": max(0.0, float(row["yhat_lower"])),
                        "y_upper": max(0.0, float(row["yhat_upper"])),
                    }
                )

        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        """Persist model + JSON sidecar.

        Note: Prophet models are themselves pickleable, so joblib handles
        the whole ``self`` (including the dict of fitted models).
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path.with_suffix(".joblib"))

        metadata = {
            "model_name": self.__class__.__name__,
            "trained_at": self._trained_at,
            "last_train_date": self._last_date.isoformat() if self._last_date else None,
            "n_groups": len(self._models),
            "groups": sorted(self._models.keys()),
            "params": {
                "date_col": self.date_col,
                "target_col": self.target_col,
                "use_br_holidays": self.use_br_holidays,
                "weekly_seasonality": self.weekly_seasonality,
                "yearly_seasonality": self.yearly_seasonality,
                "interval_width": self.interval_width,
            },
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ProphetForecaster:
        """Restore a previously saved model."""
        path = Path(path)
        model = joblib.load(path.with_suffix(".joblib"))
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is a {type(model).__name__}, expected {cls.__name__}")
        return model

    def _validate_columns(self, df: pd.DataFrame, group_col: str) -> None:
        for col in (self.date_col, group_col, self.target_col):
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col!r}")

    def _build_holidays_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the Brazilian holiday DataFrame in Prophet's expected format.

        Prophet expects two columns: ``holiday`` (label) and ``ds`` (date).
        """
        dates = pd.to_datetime(df[self.date_col])
        years = range(dates.dt.year.min(), dates.dt.year.max() + 2)  # +2 for forecast
        op_holidays = build_br_operational_holidays(years)

        holidays_df = pd.DataFrame(
            {
                "holiday": "br_operational",
                "ds": pd.to_datetime(sorted(op_holidays)),
            }
        )
        return holidays_df
