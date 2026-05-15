"""LightGBM forecaster with internal feature pipeline.

Implements :class:`~shipping_forecast.models.base.ForecastModel` using
LightGBM (via the sklearn API) as the underlying regressor. The model
internally constructs and applies the canonical feature pipeline defined
in :mod:`shipping_forecast.models.feature_config`, so callers only need
to pass raw DataFrames with date / group / target — consistent with the
Naive and SeasonalNaive baselines.

Design notes:

* The pipeline is constructed during :meth:`fit` using
  ``state_avg_volume`` computed from the training set only. The same
  pipeline (with the same statistics) is reused at :meth:`predict` time,
  so volume tier and per-state volume features are consistent.
* Days with ``is_operational == 0`` (Sundays + Brazilian holidays) are
  filtered out of training and forced to 0 at prediction time. This is
  a deterministic rule that does not need to be learned.
* Prediction uses **recursive forecasting**: one day at a time, the
  model's predictions are injected back into the working DataFrame as
  if they were the real target. This prevents lag features from being
  NaN beyond the first 7-14 days of the horizon, which would collapse
  predictions to the global mean.
* Hyperparameters use sensible defaults; Phase 6.3 will replace these
  with Optuna-tuned values.
* Prediction intervals (``y_lower`` / ``y_upper``) are not produced by
  this class. Conformal prediction wrapping comes in Phase 6.4.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd

from shipping_forecast.features import FeaturePipeline, VolumeFeatures
from shipping_forecast.models.base import ForecastModel
from shipping_forecast.models.feature_config import (
    DATE_COL,
    GROUP_COL,
    TARGET_COL,
    build_default_pipeline,
    get_feature_columns,
)

# Sensible defaults for the un-tuned baseline LightGBM. Phase 6.3 will
# replace these via Optuna. Values reflect typical defaults for tabular
# time-series regression with mild regularisation.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "verbose": -1,
    "random_state": 42,
}


class LightGBMForecaster(ForecastModel):
    """LightGBM forecaster that builds and applies its own feature pipeline.

    Attributes set after :meth:`fit`:

    * ``model_``: the trained ``lightgbm.LGBMRegressor``.
    * ``pipeline_``: the :class:`FeaturePipeline` built from training stats.
    * ``feature_names_``: ordered list of feature columns the model expects.
    * ``state_avg_volume_``: ``state -> mean daily shipments`` from train.
    * ``_last_train_date``: last date in the training data.
    * ``_trained_at``: ISO-format UTC timestamp of training.
    * ``_fitted``: ``True`` once :meth:`fit` has completed successfully.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.model_: lgb.LGBMRegressor | None = None
        self.pipeline_: FeaturePipeline | None = None
        self.feature_names_: list[str] = []
        self.state_avg_volume_: dict[str, float] = {}
        self._last_train_date: pd.Timestamp | None = None
        self._trained_at: str | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------ fit

    def fit(self, df: pd.DataFrame) -> LightGBMForecaster:
        """Train the model on a historical DataFrame."""
        self._validate_columns(df)

        state_stats = VolumeFeatures.compute_stats_from_train(
            df, group_col=GROUP_COL, target_col=TARGET_COL
        )

        pipeline = build_default_pipeline(state_stats)

        transformed = pipeline.transform(
            df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
        )

        transformed = transformed[transformed["is_operational"] == 1]
        feature_cols = get_feature_columns(transformed)
        transformed = transformed.dropna(subset=feature_cols)

        if transformed.empty:
            raise ValueError(
                "No rows left after operational filter + NaN drop. "
                "Check that the training set is long enough for lag_14."
            )

        x_train = transformed[feature_cols]
        y_train = transformed[TARGET_COL]

        model = lgb.LGBMRegressor(**self.params)
        cat_features = [c for c in feature_cols if x_train[c].dtype.name == "category"]
        model.fit(x_train, y_train, categorical_feature=cat_features or "auto")

        self.model_ = model
        self.pipeline_ = pipeline
        self.feature_names_ = feature_cols
        self.state_avg_volume_ = state_stats
        self._last_train_date = pd.Timestamp(df[DATE_COL].max())
        self._trained_at = datetime.now(UTC).isoformat()
        self._fitted = True
        return self

    # -------------------------------------------------------------- predict

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        group_col: str = GROUP_COL,
    ) -> pd.DataFrame:
        """Predict the next ``horizon`` days per group using recursive forecasting.

        At inference time the lag features for day ``t > 7`` would otherwise
        be NaN (since the true target is unknown beyond the last training
        day). We avoid that collapse by predicting one day at a time and
        injecting each prediction back into the working DataFrame as if it
        were the real target. The next iteration's lag-7 / lag-14 then see
        the predicted (rather than missing) values.

        This simulates the realistic production behaviour: a daily-retraining
        system replaces yesterday's prediction with today's actual before
        forecasting tomorrow.
        """
        if not self._fitted or self.model_ is None or self.pipeline_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        self._validate_columns(df, require_target=True)

        future = self._build_future_grid(df, horizon, group_col)
        working_df = (
            pd.concat(
                [df[[DATE_COL, group_col, TARGET_COL]], future],
                ignore_index=True,
            )
            .sort_values([group_col, DATE_COL])
            .reset_index(drop=True)
        )

        cutoff = pd.Timestamp(df[DATE_COL].max())
        future_dates = sorted(future[DATE_COL].unique())

        for current_date in future_dates:
            transformed = self.pipeline_.transform(working_df)

            day_mask = transformed[DATE_COL] == current_date
            day_rows = transformed[day_mask]
            x_day = day_rows[self.feature_names_]
            y_pred = pd.Series(
                self.model_.predict(x_day),
                index=day_rows.index,
            )

            op_mask = day_rows["is_operational"] == 1
            y_pred = y_pred.where(op_mask, 0.0).clip(lower=0.0)

            working_df.loc[working_df[DATE_COL] == current_date, TARGET_COL] = y_pred.to_numpy()

        result = (
            working_df[working_df[DATE_COL] > cutoff][[DATE_COL, group_col, TARGET_COL]]
            .copy()
            .rename(columns={TARGET_COL: "y_pred"})
            .reset_index(drop=True)
        )
        return result

    # ----------------------------------------------------------- save/load

    def save(self, path: Path) -> None:
        """Persist the model and a JSON metadata sidecar."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path.with_suffix(".joblib"))

        metadata = {
            "model_name": self.__class__.__name__,
            "trained_at": self._trained_at,
            "last_train_date": (
                self._last_train_date.date().isoformat()
                if self._last_train_date is not None
                else None
            ),
            "n_features": len(self.feature_names_),
            "feature_names": self.feature_names_,
            "n_groups": len(self.state_avg_volume_),
            "groups": sorted(self.state_avg_volume_.keys()),
            "params": self.params,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> LightGBMForecaster:
        """Restore a model previously saved with :meth:`save`."""
        path = Path(path)
        model = joblib.load(path.with_suffix(".joblib"))
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is a {type(model).__name__}, expected {cls.__name__}")
        return model

    # ----------------------------------------------------------- internals

    def _validate_columns(self, df: pd.DataFrame, require_target: bool = True) -> None:
        required = [DATE_COL, GROUP_COL]
        if require_target:
            required.append(TARGET_COL)
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col!r}")

    def _build_future_grid(self, df: pd.DataFrame, horizon: int, group_col: str) -> pd.DataFrame:
        """Build a DataFrame with one row per (future_date, group), target=NaN."""
        last_date = pd.Timestamp(df[DATE_COL].max())
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        groups = df[group_col].unique()

        future = pd.MultiIndex.from_product(
            [future_dates, groups], names=[DATE_COL, group_col]
        ).to_frame(index=False)
        future[TARGET_COL] = float("nan")
        return future
