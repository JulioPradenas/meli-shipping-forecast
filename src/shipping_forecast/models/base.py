"""Base class for forecasting models.

Defines the contract that every concrete model must satisfy:

* :meth:`fit` ingests a historical DataFrame (date + group + target).
* :meth:`predict` produces forecasts for a given horizon, returning a
  DataFrame with one row per (date, group) and a ``y_pred`` column.
  Optional ``y_lower`` / ``y_upper`` columns provide a prediction
  interval when the model supports it.

The signature is deliberately specific to time-series forecasting rather
than generic sklearn-style ``fit(X, y)`` because:

* The target depends on the temporal order, so splitting features from
  the target loses important structure.
* Models need to generate features for future dates internally
  (a Seasonal Naive needs the lag-7 from training data; LightGBM needs
  the full FeaturePipeline applied to the future dates).
* The output naturally includes metadata (date, group) that downstream
  evaluation, dashboards, and APIs all need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class ForecastModel(ABC):
    """Abstract base for any forecasting model in this project.

    Concrete subclasses must implement :meth:`fit`, :meth:`predict`,
    :meth:`save`, and :meth:`load`.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> ForecastModel:
        """Train the model on a historical DataFrame.

        Args:
            df: DataFrame with at minimum a date column, a group column
                (e.g. ``customer_state``), and a target column
                (``n_shipments``). Each concrete model documents its
                exact column requirements.

        Returns:
            ``self``, to allow chained calls like
            ``model.fit(df).predict(df, horizon=30)``.
        """

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        group_col: str = "customer_state",
    ) -> pd.DataFrame:
        """Predict the next ``horizon`` days per group.

        Args:
            df: Historical DataFrame containing data up to the day before
                the first forecast. Must contain the columns the model
                needs (date, group, target, and any features).
            horizon: Number of future days to forecast.
            group_col: Column identifying separate time series (default
                ``customer_state``).

        Returns:
            DataFrame with one row per (forecast_date, group) and columns:

            * ``shipment_date``: forecast date (datetime64).
            * ``{group_col}``: the group identifier.
            * ``y_pred``: point forecast (always present).
            * ``y_lower``: lower bound of the prediction interval (optional).
            * ``y_upper``: upper bound of the prediction interval (optional).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the model to disk.

        Creates two files:

        * ``{path}.joblib``: serialised model object.
        * ``{path}.json``: human-readable sidecar with metadata.

        Args:
            path: Path to save the model to (without extension).
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> ForecastModel:
        """Restore a model previously saved with :meth:`save`.

        Args:
            path: Same path used in :meth:`save` (without extension).

        Returns:
            A fully restored model instance ready to predict.
        """
