"""Pydantic schemas for the v1 predict API.

This module defines the contract between API callers and the service:
what a valid request looks like, what the response contains, and what
field-level constraints Pydantic enforces automatically.

Schemas defined here:
  - PredictRequest: input to POST /v1/predict
  - Prediction: one row in the response (per date x state)
  - PredictMetadata: provenance info (which alpha was used, why, etc.)
  - PredictResponse: wrapper combining predictions + metadata
  - ModelInfoResponse: schema for GET /v1/model/info (implemented in 8.4)

Validation philosophy: Pydantic handles static checks (types, ranges,
date order, horizon limits). Runtime checks that depend on the loaded
model — like "is this state known?" or "is this date in the future?" —
are enforced in the endpoint layer (8.3), not here.

Defaults that mirror Settings (alpha=0.65, cost_ratio=3.0) are intentionally
NOT duplicated in these schemas. The fields are Optional so the endpoint
can distinguish "user explicitly set this" from "user wants the default",
which feeds the alpha_source / cost_ratio_source provenance fields.
"""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictRequest(BaseModel):
    """Input payload for POST /v1/predict.

    The caller specifies a date range and optionally a subset of states.
    Cost-sensitive parameters (alpha, cost_ratio) are optional; when
    omitted, the server applies its configured defaults and reports the
    resolution via PredictMetadata.alpha_source / cost_ratio_source.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_date": "2018-09-01",
                "end_date": "2018-09-30",
                "states": ["SP", "RJ"],
                "include_intervals": True,
                "include_cost_aware": True,
                "alpha": 0.65,
                "cost_ratio": 3.0,
            }
        }
    )

    start_date: datetime.date = Field(
        ...,
        description="First date to predict, inclusive. Must be after the model's last_train_date.",
    )
    end_date: datetime.date = Field(
        ...,
        description="Last date to predict, inclusive. Must be within 90 days of start_date.",
    )
    states: list[str] | None = Field(
        default=None,
        description=(
            "Brazilian state codes (uppercase ISO 3166-2 suffix, e.g. 'SP', 'RJ'). "
            "If None or empty, predicts for all states known to the model."
        ),
    )
    include_intervals: bool = Field(
        default=True,
        description="When True, response includes lower_90/upper_90 conformal bounds.",
    )
    include_cost_aware: bool = Field(
        default=True,
        description="When True, response includes the cost-aware 'recommended' value.",
    )
    alpha: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Shift applied to predictions for cost-aware mode. Positive = sub-predict less "
            "(safer for high under-cost regimes). If None, server uses its configured default."
        ),
    )
    cost_ratio: float | None = Field(
        default=None,
        ge=0.5,
        le=10.0,
        description=(
            "Ratio of under-prediction cost to over-prediction cost. Used only when "
            "include_cost_aware=True. If None, server uses its configured default."
        ),
    )

    @model_validator(mode="after")
    def _validate_date_range(self) -> PredictRequest:
        """Enforce: end_date >= start_date AND (end_date - start_date) <= 90 days."""
        if self.end_date < self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be on or after start_date ({self.start_date})."
            )
        horizon_days = (self.end_date - self.start_date).days + 1
        if horizon_days > 90:
            raise ValueError(
                f"Horizon too long: {horizon_days} days requested, max is 90. "
                f"Use a smaller window or split into multiple requests."
            )
        return self


class Prediction(BaseModel):
    """A single prediction for a (date, state) combination.

    Fields that depend on request flags are Optional and will be None
    when the corresponding flag was False in the request:
      - lower_90 / upper_90: only when include_intervals=True
      - recommended: only when include_cost_aware=True
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "date": "2018-09-15",
                "state": "SP",
                "point": 12450.3,
                "lower_90": 10200.1,
                "upper_90": 14890.7,
                "recommended": 12947.1,
                "alpha_used": 0.65,
                "cost_ratio_used": 3.0,
            }
        }
    )

    date: datetime.date = Field(..., description="The day this prediction is for.")
    state: str = Field(..., description="Brazilian state code (e.g. 'SP', 'RJ').")
    point: float = Field(
        ...,
        ge=0.0,
        description="Calibrated point forecast (no cost shift). Non-negative.",
    )
    lower_90: float | None = Field(
        default=None,
        ge=0.0,
        description="Lower bound of the 90% conformal interval. None if intervals were not requested.",
    )
    upper_90: float | None = Field(
        default=None,
        ge=0.0,
        description="Upper bound of the 90% conformal interval. None if intervals were not requested.",
    )
    recommended: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Cost-aware recommended value: point + alpha * sigma. "
            "None if cost-aware mode was not requested."
        ),
    )
    alpha_used: float | None = Field(
        default=None,
        description="The alpha actually applied. None when include_cost_aware=False.",
    )
    cost_ratio_used: float | None = Field(
        default=None,
        description="The cost_ratio actually applied. None when include_cost_aware=False.",
    )


class PredictMetadata(BaseModel):
    """Provenance and counts for a prediction response.

    The *_source fields tell the caller whether their request-level values
    were used (``"request"``) or whether the server applied its defaults
    (``"server_default"``). This is essential for debugging: if a downstream
    consumer reports unexpected sub-prediction, the first question is
    "what alpha was actually used and where did it come from?".
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_at": "2026-05-19T14:23:01Z",
                "n_predictions": 140,
                "alpha_source": "server_default",
                "cost_ratio_source": "request",
            }
        }
    )

    predicted_at: datetime.datetime = Field(
        ...,
        description="UTC timestamp when the predictions were generated.",
    )
    n_predictions: int = Field(
        ...,
        ge=0,
        description="Total number of rows in the predictions array.",
    )
    alpha_source: Literal["request", "server_default"] = Field(
        ...,
        description=(
            "Where the alpha value came from. 'request' means the caller passed it explicitly; "
            "'server_default' means the server applied its configured default."
        ),
    )
    cost_ratio_source: Literal["request", "server_default"] = Field(
        ...,
        description="Same semantics as alpha_source, but for cost_ratio.",
    )


class PredictResponse(BaseModel):
    """Full response body for POST /v1/predict.

    The top-level wrapper combines the model identifier, the list of
    predictions (one per date x state combination), and metadata for
    provenance and debugging.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "lgbm-v1.0.0",
                "predictions": [
                    {
                        "date": "2018-09-15",
                        "state": "SP",
                        "point": 12450.3,
                        "lower_90": 10200.1,
                        "upper_90": 14890.7,
                        "recommended": 12947.1,
                        "alpha_used": 0.65,
                        "cost_ratio_used": 3.0,
                    }
                ],
                "metadata": {
                    "predicted_at": "2026-05-19T14:23:01Z",
                    "n_predictions": 1,
                    "alpha_source": "server_default",
                    "cost_ratio_source": "server_default",
                },
            }
        }
    )

    model_version: str = Field(
        ...,
        description="Identifier of the model used (from extra_metadata.version in the joblib sidecar).",
    )
    predictions: list[Prediction] = Field(
        ...,
        description="One prediction per (date, state) combination in the requested window.",
    )
    metadata: PredictMetadata = Field(
        ...,
        description="Provenance and counts for this response.",
    )


class EvaluationMetrics(BaseModel):
    """Honest out-of-sample metrics from the evaluation model.

    These metrics come from a sibling model trained only up to the
    pre-holdout cutoff (Phase 6 fold 3 boundary). The production model
    served by this API was trained on the full dataset, but its expected
    performance is reflected by these strict, leakage-free numbers.
    """

    window_start: datetime.date = Field(..., description="First day of the holdout window.")
    window_end: datetime.date = Field(..., description="Last day of the holdout window.")
    n_days: int = Field(..., gt=0, description="Number of days in the holdout window.")
    wape: float = Field(..., ge=0.0, description="Weighted absolute percentage error.")
    mae: float = Field(..., ge=0.0, description="Mean absolute error.")
    rmse: float = Field(..., ge=0.0, description="Root mean squared error.")


class ModelInfoResponse(BaseModel):
    """Response body for GET /v1/model/info.

    Surfaces the subset of model metadata that consumers care about: when
    it was trained, what dates it has seen, which states it knows, and
    what its honest holdout performance is. Implementation-internal fields
    (feature names, hyperparameters, training mode) are intentionally
    omitted to keep the API contract minimal and stable.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "lgbm-v1.0.0",
                "trained_at": "2026-05-19T00:21:15.106448+00:00",
                "last_train_date": "2018-08-31",
                "data_cutoff": "2018-08-31",
                "n_features": 23,
                "n_groups": 27,
                "groups": ["AC", "AL", "AM", "AP", "BA", "RJ", "SP"],
                "evaluation_metrics": {
                    "window_start": "2018-07-01",
                    "window_end": "2018-08-31",
                    "n_days": 62,
                    "wape": 0.5156,
                    "mae": 4.04,
                    "rmse": 14.22,
                },
                "evaluation_note": (
                    "Metrics computed by a sibling model trained only up to "
                    "2018-06-30 (Phase 6 fold 3 boundary). This production model "
                    "was trained on the full dataset..."
                ),
            }
        }
    )

    model_version: str = Field(..., description="Identifier of the deployed model.")
    trained_at: datetime.datetime = Field(
        ...,
        description="UTC timestamp when the model was trained.",
    )
    last_train_date: datetime.date = Field(
        ...,
        description="Last calendar day in the training data.",
    )
    data_cutoff: datetime.date = Field(
        ...,
        description="Upper bound on shipment_date applied at load time to drop unreliable rows.",
    )
    n_features: int = Field(..., gt=0, description="Number of features used by the model.")
    n_groups: int = Field(..., gt=0, description="Number of distinct states the model can predict.")
    groups: list[str] = Field(
        ...,
        description="The list of valid state codes that can be passed in PredictRequest.states.",
    )
    evaluation_metrics: EvaluationMetrics = Field(
        ...,
        description="Out-of-sample performance on the reserved holdout window.",
    )
    evaluation_note: str = Field(
        ...,
        description="Free-form text explaining how the metrics were obtained.",
    )
